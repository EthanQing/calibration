import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "OpenCV not installed. Please install dependencies: pip install -r requirements.txt"
    ) from e


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate multi-camera extrinsics using shared ArUco markers + per-pixel depth. "
            "World frame defaults to reference camera (camera_0)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        action="append",
        required=True,
        help=(
            "Folder containing color_*.jpg and depth_*.raw files (same timestamp). "
            "Can be provided multiple times to fuse multiple captures."
        ),
    )
    parser.add_argument(
        "--intrinsics-json",
        type=str,
        default=str(Path(__file__).with_name("cameraId_intrinsic_map.json")),
        help="Path to cameraId_intrinsic_map.json (fx,fy,cx,cy).",
    )
    parser.add_argument(
        "--ref-camera",
        type=int,
        default=0,
        help="Reference camera id used as the world frame origin.",
    )
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default="DICT_4X4_50",
        help=(
            "OpenCV aruco dictionary name, e.g. DICT_4X4_50, DICT_5X5_100, DICT_6X6_250."
        ),
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.001,
        help="Convert uint16 depth units to meters. Common: 0.001 for mm -> m.",
    )
    parser.add_argument(
        "--depth-window",
        type=int,
        default=5,
        help="Window size (odd) for median depth sampling around each corner pixel.",
    )
    parser.add_argument(
        "--min-correspondences",
        type=int,
        default=30,
        help="Minimum 3D-3D correspondences required per camera pair.",
    )
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help=(
            "Estimate pairwise camera transforms and chain them to the reference camera. "
            "Helps when a camera has no common markers directly with the reference."
        ),
    )
    parser.add_argument(
        "--ransac-iters",
        type=int,
        default=2000,
        help="RANSAC iterations for rigid alignment.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=0.015,
        help="RANSAC inlier threshold in meters.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="extrinsics_world_cam0.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def load_intrinsics_map(path: str) -> Dict[int, Intrinsics]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result: Dict[int, Intrinsics] = {}
    for key, value in data.items():
        m = re.fullmatch(r"camera_(\d+)", key)
        if not m:
            continue
        cam_id = int(m.group(1))
        result[cam_id] = Intrinsics(
            fx=float(value["fx"]),
            fy=float(value["fy"]),
            cx=float(value["cx"]),
            cy=float(value["cy"]),
        )
    return result


def get_aruco_dictionary(dict_name: str):
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(
            f"Unknown aruco dict '{dict_name}'. "
            f"Try e.g. DICT_4X4_50, DICT_5X5_100, DICT_6X6_250."
        )
    dict_id = getattr(cv2.aruco, dict_name)
    return cv2.aruco.getPredefinedDictionary(dict_id)


def parse_wh_from_name(name: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"_(\d+)x(\d+)_", name)
    if not m:
        return None
    w = int(m.group(1))
    h = int(m.group(2))
    return w, h


def load_depth_raw(path: Path, width: int, height: int) -> np.ndarray:
    depth = np.fromfile(str(path), dtype=np.uint16)
    expected = width * height
    if depth.size != expected:
        raise ValueError(
            f"Depth raw size mismatch for {path.name}: got {depth.size}, expected {expected}"
        )
    return depth.reshape((height, width))


def median_depth_at(depth_u16: np.ndarray, u: float, v: float, window: int) -> int:
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


def deproject(u: float, v: float, z_m: float, K: Intrinsics) -> np.ndarray:
    x = (u - K.cx) / K.fx * z_m
    y = (v - K.cy) / K.fy * z_m
    return np.array([x, y, z_m], dtype=np.float64)


PointKey = Tuple[int, int]  # (markerId, cornerIndex 0..3)


def detect_aruco_3d_points(
    color_path: Path,
    depth_path: Path,
    intr: Intrinsics,
    aruco_dict,
    depth_scale: float,
    depth_window: int,
) -> Dict[PointKey, np.ndarray]:
    img = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {color_path}")

    wh = parse_wh_from_name(color_path.name)
    if wh is None:
        raise ValueError(
            f"Cannot parse width/height from filename: {color_path.name} (expected _1280x720_)"
        )
    width, height = wh

    depth_u16 = load_depth_raw(depth_path, width=width, height=height)

    # OpenCV ArUco API changed around OpenCV 4.7+ and OpenCV 5:
    # - New: cv2.aruco.ArucoDetector(...).detectMarkers(image)
    # - Old: cv2.aruco.detectMarkers(image, dictionary, parameters=...)
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    elif hasattr(cv2.aruco, "DetectorParameters_create"):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        raise AttributeError("cv2.aruco.DetectorParameters is not available. Install opencv-contrib-python.")

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _rejected = detector.detectMarkers(img)
    elif hasattr(cv2.aruco, "detectMarkers"):
        corners, ids, _rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    else:
        raise AttributeError(
            "cv2.aruco.detectMarkers is not available in this OpenCV build. "
            "Please use a newer OpenCV with ArucoDetector or install opencv-contrib-python."
        )

    points: Dict[PointKey, np.ndarray] = {}
    if ids is None or len(ids) == 0:
        return points

    ids = ids.reshape(-1)
    for marker_idx, marker_id in enumerate(ids.tolist()):
        c = corners[marker_idx].reshape(4, 2)  # (4,2)
        for corner_index in range(4):
            u, v = float(c[corner_index, 0]), float(c[corner_index, 1])
            z_u16 = median_depth_at(depth_u16, u=u, v=v, window=depth_window)
            if z_u16 <= 0:
                continue
            z_m = float(z_u16) * depth_scale
            if not math.isfinite(z_m) or z_m <= 0:
                continue
            p = deproject(u=u, v=v, z_m=z_m, K=intr)
            points[(marker_id, corner_index)] = p

    return points


def estimate_rigid_transform_svd(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find R,t such that dst ~= R*src + t."""
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


def transform_points(R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (R @ pts.T).T + t.reshape(1, 3)


def ransac_rigid_transform(
    src: np.ndarray,
    dst: np.ndarray,
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

    # Precompute for speed
    indices = np.arange(n)

    for _ in range(iters):
        sample = rng.choice(indices, size=3, replace=False)
        try:
            R, t = estimate_rigid_transform_svd(src[sample], dst[sample])
        except np.linalg.LinAlgError:
            continue

        pred = transform_points(R, t, src)
        err = np.linalg.norm(pred - dst, axis=1)
        inliers = err < threshold_m
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_R = R
            best_t = t

    if best_count < max(3, min_inliers):
        raise ValueError(
            f"RANSAC failed: only {best_count} inliers (need >= {max(3, min_inliers)})"
        )

    # Refit using all inliers
    R, t = estimate_rigid_transform_svd(src[best_inliers], dst[best_inliers])
    return R, t, best_inliers


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def compose_T(T_a_from_b: np.ndarray, T_b_from_c: np.ndarray) -> np.ndarray:
    """Compose transforms: T_a_from_c = T_a_from_b @ T_b_from_c."""
    return T_a_from_b @ T_b_from_c


def bfs_world_transforms(
    ref_id: int,
    cameras: List[int],
    T_cur_from_nbr: Dict[Tuple[int, int], np.ndarray],
) -> Dict[int, np.ndarray]:
    """Compute T_world_from_cam for all reachable cameras using BFS.

    Expects edges keyed as (cur, nbr) holding T_cur_from_nbr.
    World frame is ref camera frame.
    """
    world_T: Dict[int, np.ndarray] = {ref_id: np.eye(4, dtype=np.float64)}
    queue: List[int] = [ref_id]
    in_queue = {ref_id}

    adjacency: Dict[int, List[int]] = {cid: [] for cid in cameras}
    for (cur, nbr) in T_cur_from_nbr.keys():
        adjacency.setdefault(cur, []).append(nbr)

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
            T_cur_from_n = T_cur_from_nbr[edge_key]
            world_T[nbr] = compose_T(T_world_from_cur, T_cur_from_n)
            if nbr not in in_queue:
                queue.append(nbr)
                in_queue.add(nbr)

    return world_T


class _UnionFind:
    def __init__(self, items: Iterable[int]):
        self.parent: Dict[int, int] = {int(i): int(i) for i in items}
        self.rank: Dict[int, int] = {int(i): 0 for i in items}

    def find(self, x: int) -> int:
        x = int(x)
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self) -> List[List[int]]:
        groups: Dict[int, List[int]] = {}
        for x in self.parent.keys():
            r = self.find(x)
            groups.setdefault(r, []).append(x)
        comps = [sorted(v) for v in groups.values()]
        comps.sort(key=lambda c: (-len(c), c))
        return comps


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    e = a - b
    return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))


def find_files_by_camera(data_dir: Path) -> Dict[int, Tuple[Path, Path]]:
    # Example names:
    # color_1_camera_0_1280x720_1_120908.jpg
    # depth_1_camera_0_1280x720_1_120908.raw
    rgb_paths = list(data_dir.glob("color_1_camera_*_*.jpg"))
    result: Dict[int, Tuple[Path, Path]] = {}

    cam_re = re.compile(r"color_1_camera_(\d+)_")
    for rgb in rgb_paths:
        m = cam_re.search(rgb.name)
        if not m:
            continue
        cam_id = int(m.group(1))
        depth_name = rgb.name.replace("color_", "depth_").replace(".jpg", ".raw")
        depth = data_dir / depth_name
        if depth.exists():
            result[cam_id] = (rgb, depth)

    return dict(sorted(result.items(), key=lambda x: x[0]))


def main() -> int:
    args = parse_args()

    data_dirs = [Path(p) for p in (args.data_dir or [])]
    for d in data_dirs:
        if not d.exists():
            raise SystemExit(f"data-dir not found: {d}")

    intr_map = load_intrinsics_map(args.intrinsics_json)

    per_dir_files: List[Dict[int, Tuple[Path, Path]]] = []
    all_camera_ids: List[int] = []
    for d in data_dirs:
        files = find_files_by_camera(d)
        if not files:
            raise SystemExit(
                f"No files found under {d}. Expect names like color_1_camera_0_1280x720_*.jpg and depth_1_camera_0_1280x720_*.raw"
            )
        per_dir_files.append(files)
        all_camera_ids.extend(list(files.keys()))

    camera_ids = sorted(set(all_camera_ids))
    if not camera_ids:
        raise SystemExit("No cameras found across provided data dirs")

    missing_intr = [cid for cid in camera_ids if cid not in intr_map]
    if missing_intr:
        raise SystemExit(f"Missing intrinsics for cameras: {missing_intr}")

    if int(args.ref_camera) not in camera_ids:
        raise SystemExit(f"Reference camera {args.ref_camera} not found in provided data dirs")

    aruco_dict = get_aruco_dictionary(args.aruco_dict)

    # Detect per-camera 3D corner points (per capture folder)
    per_capture_points: List[Dict[int, Dict[PointKey, np.ndarray]]] = []
    for capture_index, (d, files) in enumerate(zip(data_dirs, per_dir_files)):
        print(f"Capture {capture_index}: {d}")
        cam_points: Dict[int, Dict[PointKey, np.ndarray]] = {}
        for cam_id, (rgb, depth) in files.items():
            pts = detect_aruco_3d_points(
                color_path=rgb,
                depth_path=depth,
                intr=intr_map[cam_id],
                aruco_dict=aruco_dict,
                depth_scale=args.depth_scale,
                depth_window=args.depth_window,
            )
            cam_points[cam_id] = pts
            print(f"camera_{cam_id}: {len(pts)} 3D points from ArUco corners")
        per_capture_points.append(cam_points)

    ref_id = int(args.ref_camera)
    # Use the first capture containing the reference camera for reporting only
    ref_pts = None
    for cap in per_capture_points:
        if ref_id in cap:
            ref_pts = cap[ref_id]
            break
    if ref_pts is None:
        raise SystemExit(f"Reference camera {ref_id} missing in all captures")

    rng = np.random.default_rng(42)

    output: Dict[str, object] = {
        "world_definition": f"world = camera_{ref_id} frame",
        "depth_scale_m_per_unit": float(args.depth_scale),
        "aruco_dict": args.aruco_dict,
        "captures": [str(d) for d in data_dirs],
        "cameras": {},
    }

    # Reference camera
    output["cameras"][f"camera_{ref_id}"] = {
        "T_world_from_cam": make_T(np.eye(3), np.zeros(3)).tolist(),
        "R_world_from_cam": np.eye(3).tolist(),
        "t_world_from_cam": [0.0, 0.0, 0.0],
        "inliers": None,
        "rmse_m": 0.0,
        "num_correspondences": int(len(ref_pts)),
        "method": "reference",
    }

    # Direct-only mode and Graph mode both work across multiple captures.

    if not args.use_graph:
        # Direct-only mode: Estimate transforms from each camera -> world (ref)
        best_direct: Dict[int, Dict[str, object]] = {}
        for capture_index, cap_points in enumerate(per_capture_points):
            if ref_id not in cap_points:
                continue
            ref_pts_cap = cap_points[ref_id]
            for cam_id in camera_ids:
                if cam_id == ref_id:
                    continue
                if cam_id not in cap_points:
                    continue
                pts = cap_points[cam_id]

                common_keys = sorted(set(ref_pts_cap.keys()).intersection(pts.keys()))
                if len(common_keys) < args.min_correspondences:
                    continue

                src = np.stack([pts[k] for k in common_keys], axis=0)  # cam_i
                dst = np.stack([ref_pts_cap[k] for k in common_keys], axis=0)  # cam_ref

                try:
                    R, t, inliers = ransac_rigid_transform(
                        src=src,
                        dst=dst,
                        iters=int(args.ransac_iters),
                        threshold_m=float(args.ransac_threshold),
                        min_inliers=max(6, int(0.5 * args.min_correspondences)),
                        rng=rng,
                    )
                except ValueError:
                    continue

                pred = transform_points(R, t, src[inliers])
                err_rmse = rmse(pred, dst[inliers])
                inl = int(inliers.sum())

                prev = best_direct.get(cam_id)
                better = (
                    prev is None
                    or inl > int(prev["inliers"])
                    or (inl == int(prev["inliers"]) and float(err_rmse) < float(prev["rmse_m"]))
                )
                if better:
                    T_world_from_cam = make_T(R, t)
                    best_direct[cam_id] = {
                        "T_world_from_cam": T_world_from_cam.tolist(),
                        "R_world_from_cam": R.tolist(),
                        "t_world_from_cam": t.tolist(),
                        "inliers": inl,
                        "rmse_m": float(err_rmse),
                        "num_correspondences": int(len(common_keys)),
                        "method": "direct",
                        "capture_index": int(capture_index),
                    }

        for cam_id, entry in best_direct.items():
            output["cameras"][f"camera_{cam_id}"] = entry
            print(
                f"camera_{cam_id}: best direct from capture {entry['capture_index']} inliers={entry['inliers']} rmse={entry['rmse_m']:.4f} m"
            )

        # Report missing
        for cam_id in camera_ids:
            if cam_id == ref_id:
                continue
            if f"camera_{cam_id}" not in output["cameras"]:
                print(
                    f"camera_{cam_id}: no direct pose w.r.t camera_{ref_id} across {len(per_capture_points)} captures (try --use-graph or more overlap)"
                )
    else:
        # Graph mode: estimate pairwise transforms, then chain to reference
        print("Building pairwise camera graph...")

        T_to_from_from: Dict[Tuple[int, int], np.ndarray] = {}
        edge_stats: Dict[Tuple[int, int], Dict[str, float]] = {}

        edges_for_json: List[Dict[str, object]] = []

        best_edge_key: Dict[Tuple[int, int], Dict[str, object]] = {}

        for capture_index, cap_points in enumerate(per_capture_points):
            for i_idx, cam_i in enumerate(camera_ids):
                if cam_i not in cap_points:
                    continue
                pts_i = cap_points[cam_i]
                keys_i = set(pts_i.keys())
                for cam_j in camera_ids[i_idx + 1 :]:
                    if cam_j not in cap_points:
                        continue
                    pts_j = cap_points[cam_j]
                    common_keys = sorted(keys_i.intersection(pts_j.keys()))
                    if len(common_keys) < args.min_correspondences:
                        continue

                    src = np.stack([pts_i[k] for k in common_keys], axis=0)  # i
                    dst = np.stack([pts_j[k] for k in common_keys], axis=0)  # j

                    try:
                        R_ji, t_ji, inliers = ransac_rigid_transform(
                            src=src,
                            dst=dst,
                            iters=int(args.ransac_iters),
                            threshold_m=float(args.ransac_threshold),
                            min_inliers=max(6, int(0.5 * args.min_correspondences)),
                            rng=rng,
                        )
                    except ValueError:
                        continue

                    pred = transform_points(R_ji, t_ji, src[inliers])
                    err_rmse = rmse(pred, dst[inliers])
                    inl = int(inliers.sum())

                    T_j_from_i = make_T(R_ji, t_ji)

                    # Keep best edge per directed pair based on inliers then rmse
                    key = (cam_j, cam_i)
                    prev = best_edge_key.get(key)
                    better = (
                        prev is None
                        or inl > int(prev["inliers"])
                        or (inl == int(prev["inliers"]) and float(err_rmse) < float(prev["rmse_m"]))
                    )
                    if better:
                        best_edge_key[key] = {
                            "T": T_j_from_i,
                            "inliers": inl,
                            "rmse_m": float(err_rmse),
                            "num_correspondences": int(len(common_keys)),
                            "capture_index": int(capture_index),
                        }

        # Materialize best edges + add reverse directions
        for (to_id, from_id), info in best_edge_key.items():
            T_to_from_from[(to_id, from_id)] = info["T"]
            T_to_from_from[(from_id, to_id)] = invert_T(info["T"])
            edge_stats[(to_id, from_id)] = {
                "inliers": float(int(info["inliers"])),
                "rmse_m": float(info["rmse_m"]),
                "num_correspondences": float(int(info["num_correspondences"])),
            }
            edge_stats[(from_id, to_id)] = edge_stats[(to_id, from_id)]

            edges_for_json.append(
                {
                    "from": f"camera_{from_id}",
                    "to": f"camera_{to_id}",
                    "T_to_from_from": info["T"].tolist(),
                    "inliers": int(info["inliers"]),
                    "rmse_m": float(info["rmse_m"]),
                    "num_correspondences": int(info["num_correspondences"]),
                    "capture_index": int(info["capture_index"]),
                }
            )

        if not T_to_from_from:
            print(
                "No pairwise edges satisfied min-correspondences. "
                "Try lowering --min-correspondences (e.g. 6-12) or capture a frame with more shared markers."
            )

        # Connected components (undirected) to show overlap structure
        uf = _UnionFind(camera_ids)
        for (to_id, from_id) in T_to_from_from.keys():
            uf.union(int(to_id), int(from_id))
        comps = uf.components()
        print(f"Connected components (by shared markers): {comps}")
        output["graph"] = {
            "min_correspondences": int(args.min_correspondences),
            "ransac_threshold_m": float(args.ransac_threshold),
            "edges": edges_for_json,
            "connected_components": comps,
        }

        world_T = bfs_world_transforms(ref_id=ref_id, cameras=camera_ids, T_cur_from_nbr=T_to_from_from)

        reachable = sorted(world_T.keys())
        unreachable = sorted([c for c in camera_ids if c not in world_T])
        print(f"Reachable cameras from camera_{ref_id}: {reachable}")
        if unreachable:
            print(
                f"Unreachable cameras (no connection via shared markers): {unreachable}. "
                "You need at least some shared markers between cameras, possibly across multiple captures."
            )

        for cam_id in reachable:
            if cam_id == ref_id:
                continue
            T_world_from_cam = world_T[cam_id]
            R = T_world_from_cam[:3, :3]
            t = T_world_from_cam[:3, 3]

            # We don't have a single edge inlier count for a chained path; keep as None.
            output["cameras"][f"camera_{cam_id}"] = {
                "T_world_from_cam": T_world_from_cam.tolist(),
                "R_world_from_cam": R.tolist(),
                "t_world_from_cam": t.tolist(),
                "inliers": None,
                "rmse_m": None,
                "num_correspondences": None,
                "method": "graph",
            }

    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        # Save next to the first capture folder by default
        out_path = data_dirs[0] / out_path

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
