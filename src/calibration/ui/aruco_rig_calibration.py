from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from .calibration_controller import CameraExtrinsic, Matrix4x4, RigCalibrationResult


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


PointKey = Tuple[int, int]  # (markerId, cornerIndex 0..3)


def _parse_camera_index(camera_id: str) -> Optional[int]:
    if not camera_id.startswith("CAM-"):
        return None
    try:
        return int(camera_id.split("-", 1)[1]) - 1
    except Exception:
        return None


def _format_camera_id(camera_index: int) -> str:
    return f"CAM-{camera_index + 1:02d}"


def _parse_wh_from_name(name: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"_(\d+)x(\d+)_", name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _load_camera_params_intrinsics(path: Path) -> Dict[int, Intrinsics]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got: {type(data).__name__}")

    out: Dict[int, Intrinsics] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int):
            continue
        rgb = item.get("rgb_info", {})
        intr = rgb.get("intrinsics", {}) if isinstance(rgb, dict) else {}
        if not isinstance(intr, dict):
            continue
        out[int(idx)] = Intrinsics(
            fx=float(intr.get("fx", 0.0)),
            fy=float(intr.get("fy", 0.0)),
            cx=float(intr.get("cx", 0.0)),
            cy=float(intr.get("cy", 0.0)),
        )
    return out


def _iter_capture_dirs(base: Path) -> List[Path]:
    if not base.exists():
        return []
    dirs = [d for d in base.iterdir() if d.is_dir()]
    dirs.sort(key=lambda p: p.name)
    return dirs


def _find_latest_capture_dir(base: Path) -> Optional[Path]:
    dirs = _iter_capture_dirs(base)
    return dirs[-1] if dirs else None


def _find_files_by_camera(capture_dir: Path) -> Dict[int, Tuple[Path, Path]]:
    color_paths = list(capture_dir.glob("color_camera_CAM-*_*.png"))
    result: Dict[int, Tuple[Path, Path]] = {}

    name_re = re.compile(r"^color_camera_(CAM-\d+)_\d+x\d+_(\d{6})\.png$")
    for color_path in color_paths:
        match = name_re.match(color_path.name)
        if not match:
            continue
        camera_id = match.group(1)
        time_suffix = match.group(2)

        idx = _parse_camera_index(camera_id)
        if idx is None:
            continue

        depth_candidates = sorted(capture_dir.glob(f"depth_camera_{camera_id}_*_{time_suffix}.raw"))
        if not depth_candidates:
            continue
        result[int(idx)] = (color_path, depth_candidates[0])

    return dict(sorted(result.items(), key=lambda item: item[0]))


def _load_depth_raw(path: Path, width: int, height: int) -> np.ndarray:
    depth = np.fromfile(str(path), dtype=np.uint16)
    expected = width * height
    if depth.size != expected:
        raise ValueError(f"Depth raw size mismatch for {path.name}: got {depth.size}, expected {expected}")
    return depth.reshape((height, width))


def _median_depth_at(depth_u16: np.ndarray, u: float, v: float, window: int) -> int:
    if window <= 0:
        return 0
    if window % 2 == 0:
        window += 1

    h, w = depth_u16.shape
    x = int(round(u))
    y = int(round(v))
    r = window // 2
    x0 = max(0, x - r)
    x1 = min(w - 1, x + r)
    y0 = max(0, y - r)
    y1 = min(h - 1, y + r)
    patch = depth_u16[y0 : y1 + 1, x0 : x1 + 1].reshape(-1)
    patch = patch[patch > 0]
    if patch.size == 0:
        return 0
    return int(np.median(patch))


def _deproject(u: float, v: float, z_m: float, K: Intrinsics) -> np.ndarray:
    x = (u - K.cx) / K.fx * z_m
    y = (v - K.cy) / K.fy * z_m
    return np.array([x, y, z_m], dtype=np.float64)


def _get_aruco_dictionary(dict_name: str):
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unknown aruco dict '{dict_name}'.")
    dict_id = getattr(cv2.aruco, dict_name)
    return cv2.aruco.getPredefinedDictionary(dict_id)


def _detect_aruco_3d_points(
    color_path: Path,
    depth_path: Path,
    intr: Intrinsics,
    aruco_dict,
    *,
    depth_scale_m_per_unit: float,
    depth_window: int,
) -> Tuple[Dict[PointKey, np.ndarray], Dict[PointKey, Tuple[float, float]]]:
    img = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {color_path}")

    depth_wh = _parse_wh_from_name(depth_path.name)
    if depth_wh is None:
        raise ValueError(f"Cannot parse width/height from filename: {depth_path.name}")
    depth_w, depth_h = depth_wh
    depth_u16 = _load_depth_raw(depth_path, width=depth_w, height=depth_h)

    color_h, color_w = img.shape[:2]
    scale_x = float(depth_w) / float(color_w) if color_w else 1.0
    scale_y = float(depth_h) / float(color_h) if color_h else 1.0

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    elif hasattr(cv2.aruco, "DetectorParameters_create"):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        raise AttributeError("cv2.aruco.DetectorParameters is not available.")

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _rejected = detector.detectMarkers(img)
    elif hasattr(cv2.aruco, "detectMarkers"):
        corners, ids, _rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    else:
        raise AttributeError("cv2.aruco.detectMarkers is not available in this OpenCV build.")

    points: Dict[PointKey, np.ndarray] = {}
    pixels: Dict[PointKey, Tuple[float, float]] = {}
    if ids is None or len(ids) == 0:
        return points, pixels

    ids = ids.reshape(-1)
    for marker_idx, marker_id in enumerate(ids.tolist()):
        c = corners[marker_idx].reshape(4, 2)
        for corner_index in range(4):
            u, v = float(c[corner_index, 0]), float(c[corner_index, 1])

            u_d = u * scale_x
            v_d = v * scale_y
            z_u16 = _median_depth_at(depth_u16, u=u_d, v=v_d, window=depth_window)
            if z_u16 <= 0:
                continue
            z_m = float(z_u16) * float(depth_scale_m_per_unit)
            if not math.isfinite(z_m) or z_m <= 0:
                continue

            points[(int(marker_id), int(corner_index))] = _deproject(u=u, v=v, z_m=z_m, K=intr)
            pixels[(int(marker_id), int(corner_index))] = (u, v)

    return points, pixels


def _estimate_rigid_transform_svd(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if src.shape != dst.shape or src.shape[1] != 3:
        raise ValueError("src/dst must be Nx3")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    H = src_c.T @ dst_c
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean - (R @ src_mean)
    return R, t


def _transform_points(R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (R @ pts.T).T + t.reshape(1, 3)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    e = a - b
    return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))


def _ransac_rigid_transform(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    iters: int,
    threshold_m: float,
    min_inliers: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = src.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 correspondences")

    best_inliers = np.zeros((n,), dtype=bool)
    best_count = 0
    best_R = np.eye(3)
    best_t = np.zeros((3,))

    indices = np.arange(n)
    for _ in range(int(iters)):
        sample = rng.choice(indices, size=3, replace=False)
        try:
            R, t = _estimate_rigid_transform_svd(src[sample], dst[sample])
        except np.linalg.LinAlgError:
            continue

        pred = _transform_points(R, t, src)
        err = np.linalg.norm(pred - dst, axis=1)
        inliers = err < float(threshold_m)
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_R = R
            best_t = t

    if best_count < max(3, int(min_inliers)):
        raise ValueError(f"RANSAC failed: only {best_count} inliers (need >= {max(3, int(min_inliers))})")

    R, t = _estimate_rigid_transform_svd(src[best_inliers], dst[best_inliers])
    return R, t, best_inliers


def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _bfs_world_transforms(
    ref_id: int,
    cameras: List[int],
    T_cur_from_nbr: Dict[Tuple[int, int], np.ndarray],
) -> Dict[int, np.ndarray]:
    world_T: Dict[int, np.ndarray] = {int(ref_id): np.eye(4, dtype=np.float64)}
    queue: List[int] = [int(ref_id)]
    in_queue = {int(ref_id)}

    adjacency: Dict[int, List[int]] = {int(cid): [] for cid in cameras}
    for (cur, nbr) in T_cur_from_nbr.keys():
        adjacency.setdefault(int(cur), []).append(int(nbr))

    while queue:
        cur = queue.pop(0)
        in_queue.discard(cur)
        T_world_from_cur = world_T[cur]

        for nbr in adjacency.get(cur, []):
            if nbr in world_T:
                continue
            edge_key = (cur, nbr)
            if edge_key not in T_cur_from_nbr:
                continue
            world_T[nbr] = T_world_from_cur @ T_cur_from_nbr[edge_key]
            if nbr not in in_queue:
                queue.append(nbr)
                in_queue.add(nbr)

    return world_T


def _project(point_xyz: np.ndarray, K: Intrinsics) -> Optional[Tuple[float, float]]:
    x, y, z = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
    if z <= 0 or not math.isfinite(z):
        return None
    u = x / z * K.fx + K.cx
    v = y / z * K.fy + K.cy
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return u, v


def _to_matrix4x4(T: np.ndarray) -> Matrix4x4:
    return (
        (float(T[0, 0]), float(T[0, 1]), float(T[0, 2]), float(T[0, 3])),
        (float(T[1, 0]), float(T[1, 1]), float(T[1, 2]), float(T[1, 3])),
        (float(T[2, 0]), float(T[2, 1]), float(T[2, 2]), float(T[2, 3])),
        (float(T[3, 0]), float(T[3, 1]), float(T[3, 2]), float(T[3, 3])),
    )


def calibrate_rig_from_latest_capture(
    *,
    captures_root: Path = Path("capture_results"),
    intrinsics_path: Path = Path("configurations") / "camera_params_best.json",
    output_json_path: Optional[Path] = None,
    ref_camera_index: int = 0,
    aruco_dict: str = "DICT_7X7_250",
    depth_scale_m_per_unit: float = 0.001,
    depth_window: int = 5,
    min_correspondences: int = 12,
    ransac_iters: int = 2000,
    ransac_threshold_m: float = 0.02,
) -> RigCalibrationResult:
    capture_dir = _find_latest_capture_dir(captures_root)
    if capture_dir is None:
        raise RuntimeError(f"未找到采集数据目录 {captures_root}，请先进行一次采集。")

    intr_map = _load_camera_params_intrinsics(intrinsics_path)
    files_by_cam = _find_files_by_camera(capture_dir)
    if not files_by_cam:
        raise RuntimeError(f"采集目录中未找到可用文件: {capture_dir}")

    camera_ids = sorted(files_by_cam.keys())
    if ref_camera_index not in camera_ids:
        raise RuntimeError(f"参考相机 camera_{ref_camera_index} 不在采集目录中: {capture_dir}")

    missing_intr = [cid for cid in camera_ids if cid not in intr_map or intr_map[cid].fx == 0 or intr_map[cid].fy == 0]
    if missing_intr:
        raise RuntimeError(f"缺少内参或内参无效的相机 index: {missing_intr}")

    aruco = _get_aruco_dictionary(aruco_dict)
    rng = np.random.default_rng(42)

    cam_points: Dict[int, Dict[PointKey, np.ndarray]] = {}
    cam_pixels: Dict[int, Dict[PointKey, Tuple[float, float]]] = {}
    marker_counts: Dict[int, int] = {}

    for cam_id, (color_path, depth_path) in files_by_cam.items():
        points, pixels = _detect_aruco_3d_points(
            color_path=color_path,
            depth_path=depth_path,
            intr=intr_map[cam_id],
            aruco_dict=aruco,
            depth_scale_m_per_unit=depth_scale_m_per_unit,
            depth_window=depth_window,
        )
        cam_points[int(cam_id)] = points
        cam_pixels[int(cam_id)] = pixels
        marker_counts[int(cam_id)] = len({mid for (mid, _cidx) in points.keys()})

    ref_id = int(ref_camera_index)
    ref_points = cam_points.get(ref_id, {})
    ref_pixels = cam_pixels.get(ref_id, {})
    if not ref_points:
        raise RuntimeError(f"参考相机 {_format_camera_id(ref_id)} 未检测到 ArUco 角点，请检查标定板是否可见。")

    # Build best pairwise edges by inliers/rmse.
    best_edge: Dict[Tuple[int, int], Dict[str, object]] = {}

    for i, cam_i in enumerate(camera_ids):
        for cam_j in camera_ids[i + 1 :]:
            pts_i = cam_points.get(cam_i, {})
            pts_j = cam_points.get(cam_j, {})
            if not pts_i or not pts_j:
                continue

            common = sorted(set(pts_i.keys()) & set(pts_j.keys()))
            if len(common) < int(min_correspondences):
                continue

            src = np.stack([pts_i[k] for k in common], axis=0)
            dst = np.stack([pts_j[k] for k in common], axis=0)

            try:
                R_ji, t_ji, inliers = _ransac_rigid_transform(
                    src,
                    dst,
                    iters=int(ransac_iters),
                    threshold_m=float(ransac_threshold_m),
                    min_inliers=max(6, int(0.5 * int(min_correspondences))),
                    rng=rng,
                )
            except ValueError:
                continue

            inl = int(inliers.sum())
            pred = _transform_points(R_ji, t_ji, src[inliers])
            err_rmse = _rmse(pred, dst[inliers])

            T_j_from_i = _make_T(R_ji, t_ji)
            key = (int(cam_j), int(cam_i))
            prev = best_edge.get(key)
            better = (
                prev is None
                or inl > int(prev["inliers"])  # type: ignore[index]
                or (inl == int(prev["inliers"]) and float(err_rmse) < float(prev["rmse_m"]))  # type: ignore[index]
            )
            if better:
                best_edge[key] = {
                    "T": T_j_from_i,
                    "inliers": int(inl),
                    "rmse_m": float(err_rmse),
                    "num_correspondences": int(len(common)),
                }

            # also evaluate reverse direction to help choose the best directed edge (optional)
            # (we still add both directions later via invert_T).

    T_cur_from_nbr: Dict[Tuple[int, int], np.ndarray] = {}
    for (to_id, from_id), info in best_edge.items():
        T = info["T"]  # type: ignore[index]
        T_cur_from_nbr[(int(to_id), int(from_id))] = T
        T_cur_from_nbr[(int(from_id), int(to_id))] = _invert_T(T)

    world_T = _bfs_world_transforms(ref_id=ref_id, cameras=camera_ids, T_cur_from_nbr=T_cur_from_nbr)

    extrinsics: Dict[str, CameraExtrinsic] = {}
    transforms: Dict[str, Matrix4x4] = {}

    K_ref = intr_map[ref_id]

    for cam_id in camera_ids:
        cam_label = _format_camera_id(int(cam_id))
        marker_count = int(marker_counts.get(int(cam_id), 0))

        if cam_id not in world_T:
            extrinsics[cam_label] = CameraExtrinsic(marker_count=marker_count, reprojection_error=float("inf"))
            continue

        T_world_from_cam = world_T[cam_id]
        transforms[cam_label] = _to_matrix4x4(T_world_from_cam)

        if cam_id == ref_id:
            extrinsics[cam_label] = CameraExtrinsic(marker_count=marker_count, reprojection_error=0.0)
            continue

        pts_cam = cam_points.get(int(cam_id), {})
        common = sorted(set(pts_cam.keys()) & set(ref_points.keys()) & set(ref_pixels.keys()))
        if len(common) < 6:
            extrinsics[cam_label] = CameraExtrinsic(marker_count=marker_count, reprojection_error=float("inf"))
            continue

        R = T_world_from_cam[:3, :3]
        t = T_world_from_cam[:3, 3]
        errors_px: List[float] = []
        for key in common:
            p_cam = pts_cam[key]
            p_ref = (R @ p_cam) + t
            uv_pred = _project(p_ref, K_ref)
            if uv_pred is None:
                continue
            u_obs, v_obs = ref_pixels[key]
            du = float(uv_pred[0]) - float(u_obs)
            dv = float(uv_pred[1]) - float(v_obs)
            errors_px.append(math.hypot(du, dv))

        if not errors_px:
            reproj = float("inf")
        else:
            reproj = float(math.sqrt(sum(e * e for e in errors_px) / float(len(errors_px))))

        extrinsics[cam_label] = CameraExtrinsic(marker_count=marker_count, reprojection_error=reproj)

    # Save calibration result in the same structure as extrinsics_world_cam0.json.
    edges_for_json: List[Dict[str, object]] = []
    undirected_edges: List[Tuple[int, int]] = []
    for (to_id, from_id), info in sorted(best_edge.items(), key=lambda item: (item[0][0], item[0][1])):
        T = info["T"]  # type: ignore[index]
        edges_for_json.append(
            {
                "from": f"camera_{int(from_id)}",
                "to": f"camera_{int(to_id)}",
                "T_to_from_from": T.tolist(),
                "inliers": int(info["inliers"]),  # type: ignore[index]
                "rmse_m": float(info["rmse_m"]),  # type: ignore[index]
                "num_correspondences": int(info["num_correspondences"]),  # type: ignore[index]
                "capture_index": 0,
            }
        )
        undirected_edges.append((int(from_id), int(to_id)))

    adjacency: Dict[int, List[int]] = {int(cid): [] for cid in camera_ids}
    for a, b in undirected_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    visited: set[int] = set()
    connected_components: List[List[int]] = []
    for cid in sorted(int(c) for c in camera_ids):
        if cid in visited:
            continue
        stack = [cid]
        visited.add(cid)
        comp: List[int] = []
        while stack:
            cur = stack.pop()
            comp.append(int(cur))
            for nbr in adjacency.get(int(cur), []):
                if nbr in visited:
                    continue
                visited.add(int(nbr))
                stack.append(int(nbr))
        connected_components.append(sorted(comp))

    output: Dict[str, object] = {
        "world_definition": f"world = camera_{ref_id} frame",
        "depth_scale_m_per_unit": float(depth_scale_m_per_unit),
        "aruco_dict": str(aruco_dict),
        "captures": [str(capture_dir.name)],
        "cameras": {},
        "graph": {
            "min_correspondences": int(min_correspondences),
            "ransac_threshold_m": float(ransac_threshold_m),
            "edges": edges_for_json,
            "connected_components": connected_components,
        },
    }

    output["cameras"][f"camera_{ref_id}"] = {
        "T_world_from_cam": np.eye(4, dtype=np.float64).tolist(),
        "R_world_from_cam": np.eye(3, dtype=np.float64).tolist(),
        "t_world_from_cam": [0.0, 0.0, 0.0],
        "inliers": None,
        "rmse_m": 0.0,
        "num_correspondences": int(len(ref_points)),
        "method": "reference",
    }

    reachable = sorted(int(k) for k in world_T.keys())
    for cam_id in reachable:
        if int(cam_id) == int(ref_id):
            continue
        T_world_from_cam = world_T[int(cam_id)]
        output["cameras"][f"camera_{int(cam_id)}"] = {
            "T_world_from_cam": T_world_from_cam.tolist(),
            "R_world_from_cam": T_world_from_cam[:3, :3].tolist(),
            "t_world_from_cam": [float(x) for x in T_world_from_cam[:3, 3].tolist()],
            "inliers": None,
            "rmse_m": None,
            "num_correspondences": None,
            "method": "graph",
        }

    if output_json_path is None:
        output_json_path = Path("configurations") / f"extrinsics_world_cam{ref_id}.json"
    else:
        output_json_path = Path(output_json_path)

    try:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Keep a timestamped copy next to the capture folder for debugging/history.
    try:
        out_dir = capture_dir.parent / "calibration_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"extrinsics_world_cam{ref_id}_{stamp}.json"
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return RigCalibrationResult(extrinsics=extrinsics, relative_transforms=transforms)
