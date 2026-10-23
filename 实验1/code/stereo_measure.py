"""D435 four-dot stereo measurement. Coordinates: left IR, x right/y down/z forward."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from dashboard import Dashboard, ResultFilter


LABELS = ("TL", "TR", "BR", "BL")
EDGES = ("top", "right", "bottom", "left")


@dataclass
class Calibration:
    left_k: np.ndarray
    right_k: np.ndarray
    left_dist: np.ndarray
    right_dist: np.ndarray
    rotation: np.ndarray  # left -> right
    translation_m: np.ndarray
    width: int
    height: int

    def save(self, path: Path) -> None:
        data = {k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in vars(self).items()}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("left_k", "right_k", "left_dist", "right_dist", "rotation", "translation_m"):
            data[key] = np.asarray(data[key], dtype=np.float64)
        obj = cls(**data)
        if obj.left_k.shape != (3, 3) or obj.right_k.shape != (3, 3):
            raise ValueError("Invalid camera matrix")
        if obj.rotation.shape != (3, 3) or obj.translation_m.shape != (3,):
            raise ValueError("Invalid stereo extrinsics")
        if not 0 < np.linalg.norm(obj.translation_m) < 1:
            raise ValueError("Calibration translation must be in metres (0 < baseline < 1 m)")
        return obj

    @property
    def fundamental(self) -> np.ndarray:
        x, y, z = self.translation_m
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.linalg.inv(self.right_k).T @ cross @ self.rotation @ np.linalg.inv(self.left_k)


@dataclass
class Dot:
    center: np.ndarray
    radius: float
    circularity: float


POLARITIES = ("dark", "light")
POLARITY_ZH = {"dark": "白底黑圆", "light": "黑底白圆", "auto": "自动"}


def detect_dots(image: np.ndarray, min_area: float, max_area: float,
                min_circularity: float, roi: tuple | None = None,
                polarity: str = "dark") -> tuple[list[Dot], np.ndarray]:
    """Binary connected-component centroids, with contour-based circularity filtering.

    polarity "dark": black dots on white (inverse Otsu); "light": white dots on black.
    Either way the dots become white blobs in the returned binary image.
    """
    if polarity not in POLARITIES:
        raise ValueError(f"polarity must be one of {POLARITIES}")
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    x, y, w, h = roi or (0, 0, image.shape[1], image.shape[0])
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise ValueError("ROI must be inside both images")
    mode = cv2.THRESH_BINARY_INV if polarity == "dark" else cv2.THRESH_BINARY
    _, cropped = cv2.threshold(blurred[y:y+h, x:x+w], 0, 255, mode | cv2.THRESH_OTSU)
    binary = np.zeros_like(image)
    binary[y:y+h, x:x+w] = cropped
    count, labels, stats, centers = cv2.connectedComponentsWithStats(cropped, 8)
    dots = []
    for i in range(1, count):
        bx, by, bw, bh, area = stats[i]
        if not min_area <= area <= max_area or min(bw, bh) / max(bw, bh) < 0.45:
            continue
        if bx == 0 or by == 0 or bx + bw == w or by + bh == h:
            continue  # truncated circles are not valid measurements
        mask = (labels[by:by+bh, bx:bx+bw] == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * cv2.contourArea(contour) / max(perimeter**2, 1e-9)
        if circularity >= min_circularity:
            dots.append(Dot(centers[i] + [x, y], math.sqrt(area / math.pi), circularity))
    # Bound combinatorial search; prefer round candidates rather than large background objects.
    dots.sort(key=lambda d: d.circularity, reverse=True)
    return dots[:16], binary


@lru_cache(maxsize=32)
def _combinations(n: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(n), 4)), dtype=np.intp).reshape(-1, 4)


def hypotheses(dots: list[Dot], limit: int = 12) -> list[tuple[float, tuple[int, ...]]]:
    """Convex four-dot groups in TL, TR, BR, BL order, allowing non-rectangular targets.

    Vectorized over all 4-combinations (16 candidates = 1820 groups) for real-time use.
    """
    combos = _combinations(len(dots))
    if not len(combos):
        return []
    centers = np.array([d.center for d in dots], dtype=np.float64)
    radii_all = np.array([d.radius for d in dots], dtype=np.float64)
    p = centers[combos]
    # Image coordinates have y down: increasing angle traverses TL -> TR -> BR -> BL.
    rel = p - p.mean(axis=1, keepdims=True)
    order = np.argsort(np.arctan2(rel[..., 1], rel[..., 0]), axis=1)
    q = np.take_along_axis(p, order[..., None], axis=1)
    start = np.argmin(q.sum(axis=2), axis=1)
    order = np.take_along_axis(order, (np.arange(4) + start[:, None]) % 4, axis=1)
    q = np.take_along_axis(p, order[..., None], axis=1)
    edge = np.roll(q, -1, axis=1) - q
    # All four points are strictly convex hull vertices iff consecutive turns share one sign.
    cross = edge[..., 0] * np.roll(edge, -1, axis=1)[..., 1] - edge[..., 1] * np.roll(edge, -1, axis=1)[..., 0]
    lengths = np.linalg.norm(edge, axis=2)
    convex = (np.all(cross > 1e-6 * lengths.max(axis=1, keepdims=True)**2, axis=1)
              | np.all(cross < -1e-6 * lengths.max(axis=1, keepdims=True)**2, axis=1))
    radii = radii_all[combos]
    rmean = radii.mean(axis=1)
    nxt = np.roll(q, -1, axis=1)
    area = np.abs((q[..., 0] * nxt[..., 1] - nxt[..., 0] * q[..., 1]).sum(axis=1)) / 2
    ok = (convex & (lengths.min(axis=1) >= 3 * rmean) & (area >= 12 * rmean**2)
          & (lengths.max(axis=1) <= 6 * lengths.min(axis=1))
          & (radii.max(axis=1) <= 1.8 * radii.min(axis=1)))
    if not ok.any():
        return []
    compactness = area / np.maximum(lengths.sum(axis=1)**2, 1)
    score = radii.std(axis=1) / rmean + 0.08 / np.maximum(compactness, 0.01)
    ids = np.take_along_axis(combos, order, axis=1)
    result = [(float(s), tuple(int(i) for i in g)) for s, g in zip(score[ok], ids[ok])]
    return sorted(result)[:limit]


def undistort_pixels(points: np.ndarray, k: np.ndarray, dist: np.ndarray) -> np.ndarray:
    return cv2.undistortPoints(np.asarray(points, dtype=np.float64).reshape(-1, 1, 2), k, dist, P=k).reshape(-1, 2)


def triangulate(left: np.ndarray, right: np.ndarray, cal: Calibration) -> tuple[np.ndarray, float]:
    """General calibrated triangulation, not an assumption of identical principal points."""
    ln = cv2.undistortPoints(left.reshape(-1, 1, 2).astype(np.float64), cal.left_k, cal.left_dist).reshape(-1, 2)
    rn = cv2.undistortPoints(right.reshape(-1, 1, 2).astype(np.float64), cal.right_k, cal.right_dist).reshape(-1, 2)
    p1 = np.column_stack((np.eye(3), np.zeros(3)))
    p2 = np.column_stack((cal.rotation, cal.translation_m))
    homogeneous = cv2.triangulatePoints(p1, p2, ln.T, rn.T)
    if np.any(np.abs(homogeneous[3]) < 1e-12):
        raise ValueError("Near-zero disparity: points at infinity")
    xyz = (homogeneous[:3] / homogeneous[3]).T
    right_xyz = xyz @ cal.rotation.T + cal.translation_m
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0) or np.any(right_xyz[:, 2] <= 0):
        raise ValueError("Points are behind a camera; stereo match rejected")
    lp, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), cal.left_k, cal.left_dist)
    rp, _ = cv2.projectPoints(xyz, cv2.Rodrigues(cal.rotation)[0], cal.translation_m, cal.right_k, cal.right_dist)
    errors = np.concatenate((np.linalg.norm(lp.reshape(-1, 2) - left, axis=1),
                             np.linalg.norm(rp.reshape(-1, 2) - right, axis=1)))
    return xyz * 1000, float(np.sqrt(np.mean(errors**2)))


def measure(left_dots: list[Dot], right_dots: list[Dot], cal: Calibration,
            epipolar_px: float = 2.0, max_reprojection_px: float = 1.5,
            min_z_mm: float = 150, max_z_mm: float = 3000) -> dict | None:
    lh, rh = hypotheses(left_dots), hypotheses(right_dots)
    if not lh or not rh:
        return None
    lp = np.array([d.center for d in left_dots])
    rp = np.array([d.center for d in right_dots])
    lu = np.column_stack((undistort_pixels(lp, cal.left_k, cal.left_dist), np.ones(len(lp))))
    ru = np.column_stack((undistort_pixels(rp, cal.right_k, cal.right_dist), np.ones(len(rp))))
    lines_r = lu @ cal.fundamental.T
    lines_l = ru @ cal.fundamental
    residual = np.abs(lu @ cal.fundamental.T @ ru.T)
    distances = 0.5 * residual * (1 / np.maximum(np.linalg.norm(lines_r[:, :2], axis=1)[:, None], 1e-12)
                                  + 1 / np.maximum(np.linalg.norm(lines_l[:, :2], axis=1)[None, :], 1e-12))
    best = None
    best_score = float("inf")
    for lscore, lids in lh:
        for rscore, rids in rh:
            for matched in itertools.permutations(rids):
                epi = distances[list(lids), list(matched)]
                if epi.max() > epipolar_px:
                    continue
                # Correspondences must preserve the target's convex perimeter.
                if not cv2.isContourConvex(rp[list(matched)].astype(np.float32)):
                    continue
                left, right = lp[list(lids)], rp[list(matched)]
                try:
                    xyz, reproj = triangulate(left, right, cal)
                except ValueError:
                    continue
                if reproj > max_reprojection_px or np.any(xyz[:, 2] < min_z_mm) or np.any(xyz[:, 2] > max_z_mm):
                    continue
                # Soft depth consistency: tilted targets are allowed, not forced fronto-parallel.
                depth_spread = np.std(xyz[:, 2]) / np.mean(xyz[:, 2])
                radius_error = np.std([left_dots[a].radius / right_dots[b].radius for a, b in zip(lids, matched)])
                score = float(epi.mean() + 2 * depth_spread + radius_error + 0.15 * (lscore + rscore))
                if score >= best_score:
                    continue
                center = xyz.mean(axis=0)
                _, _, vt = np.linalg.svd(xyz - center)
                normal = vt[-1]
                plane_dist = (xyz - center) @ normal
                best_score = score
                best = {
                    "labels": list(LABELS), "left_px": left.tolist(), "right_px": right.tolist(),
                    "disparity_px": (left[:, 0] - right[:, 0]).tolist(),
                    "xyz_mm": xyz.tolist(),
                    "edges_mm": dict(zip(EDGES, np.linalg.norm(xyz - np.roll(xyz, -1, axis=0), axis=1).tolist())),
                    "plane_rms_mm": float(np.sqrt(np.mean(plane_dist**2))),
                    "plane_normal": normal.tolist(), "plane_center_mm": center.tolist(),
                    "reprojection_rms_px": reproj, "max_epipolar_error_px": float(epi.max()),
                    "match_score": score,
                }
    return best


def annotate(left: np.ndarray, right: np.ndarray, ld: list[Dot], rd: list[Dot], result: dict | None) -> np.ndarray:
    canvas = np.hstack((cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)))
    w = left.shape[1]
    colors = ((0, 220, 255), (80, 255, 80), (255, 150, 50), (255, 80, 220))
    for dots, offset in ((ld, 0), (rd, w)):
        for dot in dots:
            p = tuple(np.rint(dot.center + [offset, 0]).astype(int))
            cv2.circle(canvas, p, round(dot.radius) + 3, (130, 130, 130), 1)
    if result:
        for i, color in enumerate(colors):
            a = tuple(np.rint(result["left_px"][i]).astype(int))
            b = tuple(np.rint(np.array(result["right_px"][i]) + [w, 0]).astype(int))
            cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
            for p in (a, b):
                cv2.circle(canvas, p, 5, color, 2)
                cv2.putText(canvas, LABELS[i], (p[0]+7, p[1]-7), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
        for i, edge in enumerate(EDGES):
            cv2.putText(canvas, f"{edge}: {result['edges_mm'][edge]:.2f} mm", (15, 60 + i*28),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, colors[i], 2)
        status = f"Plane RMS {result['plane_rms_mm']:.3f} mm | Reproj RMS {result['reprojection_rms_px']:.3f} px"
        if "polarity" in result:
            status += f" | target: {'black dots on white' if result['polarity'] == 'dark' else 'white dots on black'}"
    else:
        status = f"NO VALID MATCH | candidates L={len(ld)} R={len(rd)} | Check ROI / exposure / thresholds"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (25, 25, 25), -1)
    cv2.putText(canvas, status, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    return canvas


def detect_and_measure(left: np.ndarray, right: np.ndarray, cal: Calibration, args,
                       order: tuple[str, ...]):
    """Try target polarities in order and keep the first one that yields a valid stereo match.

    Returns (result, left_dots, right_dots, left_binary, right_binary, polarity). When none
    matches, the dots / binaries of the first polarity are returned for display.
    """
    first = None
    for polarity in order:
        ld, lb = detect_dots(left, args.min_area, args.max_area, args.circularity, args.roi, polarity)
        rd, rb = detect_dots(right, args.min_area, args.max_area, args.circularity, args.roi, polarity)
        result = measure(ld, rd, cal, args.epipolar_px, args.reprojection_px, args.min_z_mm, args.max_z_mm)
        if result:
            result["polarity"] = polarity
            return result, ld, rd, lb, rb, polarity
        first = first or (None, ld, rd, lb, rb, polarity)
    return first


def save_capture(root: Path, left: np.ndarray, right: np.ndarray, cal: Calibration,
                 result: dict | None, view: np.ndarray, lb: np.ndarray, rb: np.ndarray,
                 metadata: dict, color: np.ndarray | None = None,
                 filtered: dict | None = None, charts: np.ndarray | None = None) -> Path:
    folder = root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    images = [("left.png", left), ("right.png", right), ("annotated.png", view),
              ("left_binary.png", lb), ("right_binary.png", rb)]
    if color is not None:
        images.append(("color.png", color))
    if charts is not None:
        images.append(("charts.png", charts))
    for name, image in images:
        if not cv2.imwrite(str(folder / name), image):
            raise OSError(f"Cannot write {folder / name}")
    cal.save(folder / "calibration.json")
    payload = {"timestamp": datetime.now().isoformat(), "valid": result is not None,
               "coordinate_system": "left IR: x right, y down, z forward; mm", "capture": metadata,
               "measurement": result, "filtered": filtered}
    (folder / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if result:
        with (folder / "edges.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(("edge", "from", "to", "length_mm", "filtered_length_mm", "filtered_std_mm"))
            for i, edge in enumerate(EDGES):
                writer.writerow((edge, LABELS[i], LABELS[(i+1) % 4], result["edges_mm"][edge],
                                 filtered["edges_mm"][edge] if filtered else "",
                                 filtered["edges_std_mm"][edge] if filtered else ""))
    return folder


class RealSenseSource:
    def __init__(self, args: argparse.Namespace):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("Install pyrealsense2 first: pip install -r requirements.txt") from exc
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if args.serial:
            config.enable_device(args.serial)
        for index in (1, 2):
            config.enable_stream(rs.stream.infrared, index, args.width, args.height, rs.format.y8, args.fps)
        if args.color:
            config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        self.started = False
        try:
            profile = self.pipeline.start(config)
            self.started = True
            device = profile.get_device()
            sensor = device.first_depth_sensor()
            if not sensor.supports(rs.option.emitter_enabled):
                raise RuntimeError("Device cannot disable IR emitter; refusing contaminated acquisition")
            self.sensor = sensor
            self.old_emitter = sensor.get_option(rs.option.emitter_enabled)
            sensor.set_option(rs.option.emitter_enabled, 0)
            if args.exposure is not None:
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.exposure, args.exposure)
            if args.gain is not None:
                sensor.set_option(rs.option.gain, args.gain)
            left = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            right = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            li, ri = left.get_intrinsics(), right.get_intrinsics()

            def intrinsics(intr):
                coeff = np.asarray(intr.coeffs, dtype=float)
                # OpenCV's standard 5-term model cannot represent inverse Brown or fisheye.
                if np.any(coeff != 0) and intr.model != rs.distortion.brown_conrady:
                    raise RuntimeError(f"Unsupported nonzero IR distortion: {intr.model}; rectify images first")
                return np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]]), coeff

            lk, ld = intrinsics(li)
            rk, rd = intrinsics(ri)
            extr = left.get_extrinsics_to(right)
            self.cal = Calibration(lk, rk, ld, rd, np.array(extr.rotation).reshape(3, 3, order="F"),
                                   np.array(extr.translation), li.width, li.height)
            self.metadata = {"device": device.get_info(rs.camera_info.name),
                             "serial": device.get_info(rs.camera_info.serial_number),
                             "fps": args.fps, "emitter_enabled": False, "depth_stream": False,
                             "average_frames": args.average}
            print(json.dumps({**self.metadata, "baseline_mm": np.linalg.norm(self.cal.translation_m)*1000}, indent=2))
            for _ in range(args.warmup):
                self.pipeline.wait_for_frames(10000)
        except Exception:
            self.close()
            raise

    def read(self, average: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        left, right = [], []
        color = None
        for _ in range(average):
            frames = self.pipeline.wait_for_frames(10000)
            l = frames.get_infrared_frame(1)
            r = frames.get_infrared_frame(2)
            if not l or not r:
                raise RuntimeError("Missing synchronized IR1/IR2 frames")
            if abs(l.get_timestamp() - r.get_timestamp()) > 2:
                raise RuntimeError("IR1/IR2 timestamp mismatch exceeds 2 ms")
            left.append(np.asanyarray(l.get_data()).copy())
            right.append(np.asanyarray(r.get_data()).copy())
            c = frames.get_color_frame()
            if c:
                color = np.asanyarray(c.get_data()).copy()
        return (np.rint(np.mean(left, axis=0)).astype(np.uint8),
                np.rint(np.mean(right, axis=0)).astype(np.uint8), color)

    def close(self):
        if self.started:
            try:
                if hasattr(self, "old_emitter"):
                    self.sensor.set_option(self.rs.option.emitter_enabled, self.old_emitter)
            finally:
                self.pipeline.stop()
                self.started = False


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D435 four-dot stereo measurement (no depth stream)")
    p.add_argument("--offline", type=Path, help="Saved folder containing left.png, right.png, calibration.json")
    p.add_argument("--calibration", type=Path, help="Override factory/offline calibration with chessboard calibration JSON")
    p.add_argument("--serial", help="Select a particular device")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--average", type=int, default=1, help="Synchronized frame pairs per measurement; keep target stationary")
    p.add_argument("--exposure", type=float, help="Manual IR exposure in microseconds")
    p.add_argument("--gain", type=float)
    p.add_argument("--color", action="store_true", help="Enable RGB reference only; never used for measurement")
    p.add_argument("--min-area", type=float, default=15)
    p.add_argument("--max-area", type=float, default=800)
    p.add_argument("--circularity", type=float, default=.65)
    p.add_argument("--epipolar-px", type=float, default=2)
    p.add_argument("--reprojection-px", type=float, default=1.5)
    p.add_argument("--min-z-mm", type=float, default=150)
    p.add_argument("--max-z-mm", type=float, default=3000)
    p.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="Same ROI for both IR images; include the disparity shift")
    p.add_argument("--output", type=Path, default=Path("output"))
    p.add_argument("--headless", action="store_true", help="No windows; saves final frame automatically")
    p.add_argument("--frames", type=int, default=0, help="Measurement count; 0 = continuous (headless defaults to 1)")
    p.add_argument("--live", nargs="?", const="text", choices=("text", "jsonl"),
                   help="Print every measurement to stdout (text or JSON Lines); headless runs continuously")
    p.add_argument("--polarity", choices=("auto",) + POLARITIES, default="auto",
                   help="Target type: auto (default), dark = black dots on white, light = white dots on black")
    p.add_argument("--filter-window", type=int, default=15,
                   help="Frames in the sliding trimmed-mean filter of 3-D centres; 1 disables filtering")
    p.add_argument("--filter-jump-mm", type=float, default=5,
                   help="Centre jump treated as outlier; 3 consistent jumps restart the filter (target moved)")
    p.add_argument("--no-charts", action="store_true", help="Do not open the live chart window")
    p.add_argument("--chart-seconds", type=float, default=10, help="History length of the live time-series chart")
    p.add_argument("--font", help="中文字体文件路径（图表窗口用），默认自动查找系统字体")
    return p


def live_line(result: dict | None, ld: list, rd: list, fmt: str, filtered: dict | None = None) -> str:
    stamp = datetime.now().isoformat(timespec="milliseconds")
    if fmt == "jsonl":
        record = {"time": stamp, "valid": result is not None, "candidates": [len(ld), len(rd)]}
        if result:
            record.update(polarity=result.get("polarity"), edges_mm=result["edges_mm"], center_z_mm=result["plane_center_mm"][2],
                          plane_rms_mm=result["plane_rms_mm"],
                          reprojection_rms_px=result["reprojection_rms_px"])
        if filtered:
            record["filtered"] = {k: filtered[k] for k in
                                  ("edges_mm", "edges_std_mm", "samples", "stale", "outlier") if k in filtered}
        return json.dumps(record, ensure_ascii=False)
    if not result:
        held = ""
        if filtered:
            held = " | 保持 " + "  ".join(f"{n} {filtered['edges_mm'][n]:.2f}" for n in EDGES)
        return f"{stamp}  无有效匹配（候选 左={len(ld)} 右={len(rd)}）{held}"
    source = filtered or result
    edges = "  ".join(f"{name} {source['edges_mm'][name]:7.2f}" for name in EDGES)
    tag = f" | {POLARITY_ZH[result['polarity']]}" if result.get("polarity") else ""
    if filtered:
        tag += f" | 滤波 {filtered['samples']}/{filtered['window']}" + (" 离群已忽略" if filtered.get("outlier") else "")
    return (f"{stamp}  {edges} mm | Z {result['plane_center_mm'][2]:.1f} mm"
            f" | 重投影 {result['reprojection_rms_px']:.2f} px{tag}")


def smoother_raw(result: dict | None) -> dict | None:
    """Raw result in the filtered-result shape, for charts when filtering is disabled."""
    if not result:
        return None
    return {**result, "edges_std_mm": dict.fromkeys(EDGES, 0.), "samples": 1, "window": 1,
            "rejected_total": 0, "stale": False, "outlier": False}


def main() -> int:
    p = parser()
    args = p.parse_args()
    if args.filter_window < 1 or args.filter_jump_mm <= 0 or args.chart_seconds <= 0:
        p.error("Require filter-window >= 1, filter-jump-mm > 0, chart-seconds > 0")
    if args.average < 1 or args.average > 32 or args.frames < 0 or args.warmup < 0:
        p.error("Require 1 <= average <= 32, frames >= 0, warmup >= 0")
    if not 0 < args.min_area < args.max_area or not 0 < args.circularity <= 1:
        p.error("Invalid area / circularity thresholds")
    if not 0 < args.min_z_mm < args.max_z_mm or min(args.epipolar_px, args.reprojection_px) <= 0:
        p.error("Invalid depth / pixel error thresholds")
    source = None
    try:
        if args.offline:
            cal = Calibration.load(args.offline / "calibration.json")
            left = cv2.imread(str(args.offline / "left.png"), cv2.IMREAD_GRAYSCALE)
            right = cv2.imread(str(args.offline / "right.png"), cv2.IMREAD_GRAYSCALE)
            if left is None or right is None:
                raise ValueError("Offline folder must contain readable left.png and right.png")
            metadata = {"source": str(args.offline), "mode": "offline"}
        else:
            source = RealSenseSource(args)
            cal, metadata = source.cal, source.metadata
        if args.calibration:
            cal = Calibration.load(args.calibration)
            metadata = {**metadata, "calibration_source": str(args.calibration)}
        if not args.headless:
            cv2.namedWindow("D435 stereo | S save | B binary | Q quit", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("D435 stereo | S save | B binary | Q quit", 1600, 600)
        print("S: save capture/results | B: toggle binary | Q/Esc: quit")
        smoother = ResultFilter(args.filter_window, args.filter_jump_mm)
        charts = None
        if not args.no_charts:
            try:
                charts = Dashboard(args.chart_seconds, args.font)
            except ValueError as exc:
                print(f"WARNING: charts disabled: {exc}")
        chart_window = "Measurement charts | S save | Q quit"
        if charts and not args.headless:
            cv2.namedWindow(chart_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(chart_window, 1280, 800)
        count, show_binary, result, filtered = 0, False, None, None
        last_polarity = "dark"
        limit = args.frames or (1 if (args.headless and not args.live) or args.offline else 0)
        while True:
            color = None
            if source:
                left, right, color = source.read(args.average)
            if left.shape != (cal.height, cal.width) or right.shape != left.shape:
                raise ValueError("Image size does not match calibration; do not resize measurement images")
            if args.polarity == "auto":
                # Try the polarity that worked last first, so a stable target costs one pass.
                order = (last_polarity,) + tuple(p for p in POLARITIES if p != last_polarity)
            else:
                order = (args.polarity,)
            result, ld, rd, lb, rb, polarity = detect_and_measure(left, right, cal, args, order)
            if result:
                last_polarity = polarity
            filtered = smoother.update(result) if args.filter_window > 1 else None
            view = annotate(left, right, ld, rd, result)
            count += 1
            if args.live:
                print(live_line(result, ld, rd, args.live, filtered), flush=True)
            chart_image = None
            # Without filtering the charts show raw values with zero spread.
            shown = filtered or smoother_raw(result)
            if charts:
                charts.update(result, shown)
                if not args.headless:
                    chart_image = charts.render(result, shown)
                    cv2.imshow(chart_window, chart_image)
            key = -1
            if not args.headless:
                cv2.imshow("D435 stereo | S save | B binary | Q quit", np.hstack((lb, rb)) if show_binary else view)
                if color is not None:
                    cv2.imshow("RGB reference (not used for measurement)", color)
                key = cv2.waitKey(0 if args.offline else 1) & 0xFF
            if key == ord("b"):
                show_binary = not show_binary
                if args.offline:
                    continue
            finished = bool(limit and count >= limit)
            if key == ord("s") or finished:
                if charts and chart_image is None:
                    chart_image = charts.render(result, shown)
                folder = save_capture(args.output, left, right, cal, result, view, lb, rb, metadata, color,
                                      filtered, chart_image)
                print(f"Saved: {folder}")
                summary = {"raw": result["edges_mm"]} if result else {"valid": False, "reason": "No valid four-dot match"}
                if filtered:
                    summary["filtered"] = filtered["edges_mm"]
                print(json.dumps(summary, indent=2))
            if key in (ord("q"), 27) or finished:
                break
        return 0 if result else 2
    except (RuntimeError, ValueError, OSError, cv2.error) as exc:
        print(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if source:
            source.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
