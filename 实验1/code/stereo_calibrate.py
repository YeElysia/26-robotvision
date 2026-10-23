"""Chessboard calibration of synchronized, unrectified D435 IR1 / IR2 images."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from dashboard import chinese_font, render_panel_text
from stereo_measure import Calibration, RealSenseSource


def board_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    if min(cols, rows) < 3 or not np.isfinite(square_m) or square_m <= 0:
        raise ValueError("Require at least 3x3 inner corners and a positive square size")
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square_m
    return points


def find_corners(image: np.ndarray, pattern: tuple[int, int], fast: bool = False) -> np.ndarray | None:
    # Keep full resolution: downsampling can erase small squares in distant IR boards.
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    if not fast:
        flags |= cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(image, pattern, flags)
    if not found:
        # Local contrast helps IR illumination gradients; the classic detector also
        # handles some boards that SB misses. Neither path changes saved images.
        enhanced = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(image)
        found, corners = cv2.findChessboardCorners(
            enhanced, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    # Some OpenCV versions return (N, 2) instead of (N, 1, 2); normalize the shape.
    corners = corners.astype(np.float32).reshape(-1, 1, 2)
    # Do not let a fixed subpixel window span adjacent corners on small boards.
    spacing = np.linalg.norm(np.diff(corners.reshape(pattern[1], pattern[0], 2), axis=1), axis=2)
    radius = max(1, min(5, int(np.median(spacing) / 3)))
    return cv2.cornerSubPix(image, corners, (radius, radius), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .01))


def match_order(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Reverse right corners when the detector returned the board rotated by 180 degrees."""
    lv = left[-1, 0] - left[0, 0]
    rv = right[-1, 0] - right[0, 0]
    return right[::-1].copy() if float(lv @ rv) < 0 else right


ANGLE_NAMES = ("up-left", "up", "up-right", "left", "front", "right",
               "down-left", "down", "down-right")
ANGLE_ZH = ("左上倾斜", "向上倾斜", "右上倾斜", "向左倾斜", "正面", "向右倾斜",
            "左下倾斜", "向下倾斜", "右下倾斜")


def board_pose(objects: np.ndarray, corners: np.ndarray, k: np.ndarray,
               dist: np.ndarray) -> dict:
    ok, rv, tv = cv2.solvePnP(objects, corners, k, dist)
    if not ok or tv[2, 0] <= 0:
        raise ValueError("Cannot estimate forward-facing board pose")
    normal = cv2.Rodrigues(rv)[0][:, 2]
    if normal[2] < 0:
        normal = -normal
    yaw = float(np.degrees(np.arctan2(normal[0], normal[2])))
    pitch = float(np.degrees(np.arctan2(normal[1], normal[2])))
    col = 0 if yaw < -15 else 2 if yaw > 15 else 1
    row = 0 if pitch < -15 else 2 if pitch > 15 else 1
    projected = cv2.projectPoints(objects, rv, tv, k, dist)[0]
    error = float(np.sqrt(np.mean(np.sum((projected - corners)**2, axis=2))))
    center = (cv2.Rodrigues(rv)[0] @ objects.mean(axis=0) + tv.ravel())
    return {"yaw_deg": yaw, "pitch_deg": pitch, "distance_m": float(center[2]),
            "center_px": corners.reshape(-1, 2).mean(axis=0).tolist(),
            "bin": row * 3 + col, "pose_rms_px": error}


class Coverage:
    def __init__(self, target: int):
        self.target = target
        self.poses = []

    @property
    def counts(self):
        return [sum(p["bin"] == i for p in self.poses) for i in range(9)]

    def novel(self, pose: dict, size: tuple[int, int]) -> bool:
        for old in self.poses:
            delta = np.linalg.norm((np.array(pose["center_px"]) - old["center_px"]) / size)
            angle = np.hypot(pose["yaw_deg"] - old["yaw_deg"], pose["pitch_deg"] - old["pitch_deg"])
            depth = abs(pose["distance_m"] / old["distance_m"] - 1)
            if delta < .08 and angle < 8 and depth < .12:
                return False
        return True

    def report(self):
        counts = self.counts
        return {"target_per_bin": self.target, "counts": dict(zip(ANGLE_NAMES, counts)),
                "progress_percent": 100 * sum(min(n, self.target) for n in counts) / (9 * self.target),
                "poses": self.poses,
                "note": "Pose coverage is not an accuracy guarantee; live poses use factory intrinsics."}


def write_calibration(args, objects, left, right, size, used):
    cal, report = calibrate_pairs(objects, left, right, size)
    coverage = Coverage(args.target_per_angle)
    coverage.poses = [board_pose(objects, corners, cal.left_k, cal.left_dist) for corners in left]
    report.update(inner_corners=[args.cols, args.rows], square_mm=args.square_mm,
                  used_pairs=list(used), angle_coverage=coverage.report())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cal.save(args.output)
    report_path = args.output.with_name(args.output.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def calibrate_pairs(object_points: np.ndarray, left: list[np.ndarray], right: list[np.ndarray],
                    size: tuple[int, int]) -> tuple[Calibration, dict]:
    if len(left) != len(right) or len(left) < 10:
        raise ValueError("At least 10 valid synchronized chessboard pairs are required")
    objects = [object_points] * len(left)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
    lrms, lk, ld, lrv, ltv = cv2.calibrateCamera(objects, left, size, None, None, criteria=criteria)
    rrms, rk, rd, rrv, rtv = cv2.calibrateCamera(objects, right, size, None, None, criteria=criteria)
    rms, lk, ld, rk, rd, rotation, translation, _, _ = cv2.stereoCalibrate(
        objects, left, right, lk, ld, rk, rd, size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
    cal = Calibration(lk, rk, ld.ravel(), rd.ravel(), rotation, translation.ravel(), *size)
    baseline = float(np.linalg.norm(cal.translation_m))
    if not np.isfinite(rms) or not 0 < baseline < 1:
        raise ValueError("Invalid calibration; check board size, corner ordering and image pairs")

    def errors(observations, k, dist, rotations, translations):
        return [float(np.sqrt(np.mean(np.sum((
            cv2.projectPoints(object_points, rv, tv, k, dist)[0].reshape(-1, 2)
            - observed.reshape(-1, 2))**2, axis=1))))
            for observed, rv, tv in zip(observations, rotations, translations)]

    report = {"pairs": len(left), "left_rms_px": float(lrms), "right_rms_px": float(rrms),
              "stereo_rms_px": float(rms), "baseline_mm": baseline * 1000,
              "left_per_view_rms_px": errors(left, lk, ld, lrv, ltv),
              "right_per_view_rms_px": errors(right, rk, rd, rrv, rtv)}
    return cal, report


def collect(args: argparse.Namespace, pattern: tuple[int, int]):
    font = chinese_font(args.font)
    source = None
    window = "Stereo calibration | A auto | S capture | C solve | Q finish"
    objects = board_points(*pattern, args.square_mm / 1000)
    coverage = Coverage(args.target_per_angle)
    left_points, right_points, used = [], [], []
    auto = not args.manual
    previous = None
    stable_since = None
    last_saved = -float("inf")
    status = "请调整到未完成的角度，保持静止后自动采集"
    future = None
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        source = RealSenseSource(args)
        args.images.mkdir(parents=True, exist_ok=True)
        print("A: toggle auto | S: manual capture | C: preview calibration | Q/Esc: finish and save")
        print("Angle labels describe board NORMAL in left IR coordinates, not hand motion.")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1600, 850)
        while True:
            left, right, _ = source.read(1)
            lc, rc = find_corners(left, pattern, fast=True), find_corners(right, pattern, fast=True)
            now = time.monotonic()
            if future is not None and future.done():
                try:
                    _, report = future.result()
                    status = f"标定预览：双目 RMS {report['stereo_rms_px']:.3f} 像素（{report['pairs']} 对）"
                except (ValueError, cv2.error) as exc:
                    status = f"预览计算失败：{exc}"
                future = None
            ready = lc is not None and rc is not None
            if ready:
                rc = match_order(lc, rc)
            pose = None
            if ready:
                try:
                    pose = board_pose(objects, lc, source.cal.left_k, source.cal.left_dist)
                except (ValueError, cv2.error):
                    ready = False
            current = np.vstack((lc, rc)) if ready else None
            if current is not None and previous is not None:
                movement = float(np.max(np.linalg.norm(current - previous, axis=2)))
                if movement <= args.stable_px:
                    if stable_since is None:
                        stable_since = now
                else:
                    stable_since = None
            else:
                stable_since = None
            previous = current
            stable = stable_since is not None and now - stable_since >= args.hold_seconds
            novel = pose is not None and coverage.novel(pose, (left.shape[1], left.shape[0]))
            needed = pose is not None and coverage.counts[pose['bin']] < args.target_per_angle
            views = []
            for image, corners in ((left, lc), (right, rc)):
                view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                if corners is not None:
                    cv2.drawChessboardCorners(view, pattern, corners, True)
                views.append(view)
            # Resize display only; saved images and corner coordinates stay full resolution.
            pair_view = np.hstack(views)
            pair_view = cv2.resize(pair_view, (1280, round(pair_view.shape[0] * 1280 / pair_view.shape[1])))
            panel = np.zeros((310, 1280, 3), np.uint8)
            labels = []
            def text(message, x, y, color=(220, 220, 220), scale=.55):
                labels.append((message, x, y, color))
            progress = coverage.report()['progress_percent']
            text(f"{'自动采集' if auto else '手动采集'} | 已保存 {len(used)} 对 | 角度完成进度 {progress:.0f}%", 15, 25)
            if pose is not None:
                text(f"偏航 {pose['yaw_deg']:+.1f}°  俯仰 {pose['pitch_deg']:+.1f}° | 距离 {pose['distance_m']:.2f} 米 | {ANGLE_ZH[pose['bin']]}", 15, 50)
                text(f"{'已稳定' if stable else '请保持静止'} | {'新姿态' if novel else '请移动、倾斜或改变距离'} | 姿态 RMS {pose['pose_rms_px']:.2f} 像素", 15, 75)
            else:
                text("未能估计姿态：请让左右相机都看到完整、清晰的棋盘", 15, 50, (0, 140, 255))
                text(f"左图：{'已检测' if lc is not None else '未检测'}  右图：{'已检测' if rc is not None else '未检测'} | 需要 {pattern[0]}×{pattern[1]} 个内角点", 15, 75)
            for i, (name, count) in enumerate(zip(ANGLE_NAMES, coverage.counts)):
                x, y = 15 + (i % 3) * 230, 90 + (i // 3) * 53
                color = (60, 190, 60) if count >= args.target_per_angle else (90, 90, 90)
                if pose is not None and i == pose['bin']:
                    cv2.rectangle(panel, (x-3, y-3), (x+220, y+45), (0, 230, 255), 2)
                cv2.rectangle(panel, (x, y+25), (x+210, y+40), (40, 40, 40), -1)
                cv2.rectangle(panel, (x, y+25), (x+round(210*min(count/args.target_per_angle, 1)), y+40), color, -1)
                text(f"{ANGLE_ZH[i]}：{count}/{args.target_per_angle}", x, y+17)
            missing = [name for name, count in zip(ANGLE_ZH, coverage.counts) if count < args.target_per_angle]
            text("建议采集：" + (missing[0] if missing else "所有角度已完成"), 730, 110)
            text("正面：偏航、俯仰均不超过 ±15°", 730, 140)
            text("其他分区：对应方向倾斜超过 15°", 730, 165)
            text(f"检测目标：{pattern[0]}×{pattern[1]} 个内角点", 730, 195, (0, 180, 255))
            text("D：保存诊断图像；进度不代表精度", 730, 220)
            text(status[:140], 15, 267)
            text("A 自动开关 | S 手动采集 | C 标定预览 | Q 结束并保存 | 角度为出厂内参估计的棋盘法向", 15, 295)
            panel = render_panel_text(panel, labels, font)
            view = np.vstack((pair_view, panel))
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                auto = not auto
            if key == ord("d"):
                folder = args.images / "diagnostics" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                folder.mkdir(parents=True)
                for name, image in (("left", left), ("right", right), ("preview", view)):
                    if not cv2.imwrite(str(folder / f"{name}.png"), image):
                        raise OSError(f"Cannot write diagnostic {name}")
                status = f"诊断图像已保存：{folder}"
                print(status)
            if key == ord("c") and future is None:
                if len(used) >= 10:
                    future = executor.submit(calibrate_pairs, objects, list(left_points), list(right_points),
                                             (left.shape[1], left.shape[0]))
                    status = "正在后台计算标定预览……"
                else:
                    status = "至少采集 10 对图像后才能预览标定"
            automatic = (auto and ready and stable and novel and needed
                         and now - last_saved >= args.capture_interval)
            if (key == ord("s") and ready) or automatic:
                folder = args.images / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                folder.mkdir()
                for name, image in (("left", left), ("right", right)):
                    if not cv2.imwrite(str(folder / f"{name}.png"), image):
                        raise OSError(f"Cannot write {folder / name}")
                (folder / "capture.json").write_text(
                    json.dumps({**source.metadata, "board_pose": pose,
                                "inner_corners": list(pattern), "square_mm": args.square_mm}, indent=2), encoding="utf-8")
                coverage.poses.append(pose)
                left_points.append(lc.copy())
                right_points.append(rc.copy())
                used.append(str(folder))
                last_saved = now
                stable_since = None
                status = f"已保存 {len(used)} 对 | 最近采集：{ANGLE_ZH[pose['bin']]}"
                (args.images / "capture_progress.json").write_text(
                    json.dumps(coverage.report(), indent=2), encoding="utf-8")
                print(f"Saved: {folder}")
                if all(n >= args.target_per_angle for n in coverage.counts) and len(used) >= 10:
                    print("Capture target complete. Computing and saving final calibration automatically...")
                    break
    finally:
        if source:
            source.close()
        cv2.destroyAllWindows()
        executor.shutdown(wait=True)
    return left_points, right_points, (source.cal.width, source.cal.height), used, []


def load_pairs(root: Path, pattern: tuple[int, int]):
    left, right, used, skipped = [], [], [], []
    size = None
    # Each subfolder contains one synchronized pair, just like measurement captures.
    paths = sorted(root.glob("*/left.png"))
    for path in paths:
        l = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(str(path.with_name("right.png")), cv2.IMREAD_GRAYSCALE)
        if l is None or r is None:
            raise ValueError(f"Unreadable image pair: {path.parent}")
        current_size = (l.shape[1], l.shape[0])
        if l.shape != r.shape or (size is not None and current_size != size):
            raise ValueError(f"All images must have the same original resolution: {path.parent}")
        size = current_size
        lc, rc = find_corners(l, pattern), find_corners(r, pattern)
        if lc is None or rc is None:
            skipped.append(str(path.parent))
            print(f"Skipped (board not found in both images): {path.parent}")
            continue
        rc = match_order(lc, rc)
        left.append(lc)
        right.append(rc)
        used.append(str(path.parent))
    return left, right, size, used, skipped


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D435 stereo chessboard calibration (12x10 squares, 20 mm)")
    p.add_argument("--images", type=Path, default=Path("calibration_images"))
    p.add_argument("--offline", action="store_true", help="Calibrate existing image pairs without a camera or GUI")
    p.add_argument("--cols", type=int, default=11, help="INNER corner columns, not square count (default 11)")
    p.add_argument("--rows", type=int, default=9, help="INNER corner rows, not square count (default 9)")
    p.add_argument("--square-mm", type=float, default=20)
    p.add_argument("--output", type=Path, default=Path("calibration.json"))
    p.add_argument("--serial")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--exposure", type=float)
    p.add_argument("--gain", type=float)
    p.add_argument("--font", help="中文字体文件路径（TTF/TTC/OTF），默认自动查找系统字体")
    p.add_argument("--manual", action="store_true", help="Start with automatic capture disabled")
    p.add_argument("--target-per-angle", type=int, default=3, help="Required pairs in each of 9 angle bins")
    p.add_argument("--hold-seconds", type=float, default=.6, help="Stable detection duration before automatic capture")
    p.add_argument("--stable-px", type=float, default=2., help="Maximum inter-detection corner motion for stability")
    p.add_argument("--capture-interval", type=float, default=1.5, help="Minimum seconds between automatic captures")
    p.set_defaults(color=False, average=1)
    return p


def main() -> int:
    p = parser()
    args = p.parse_args()
    try:
        objects = board_points(args.cols, args.rows, args.square_mm / 1000)
        if args.warmup < 0:
            raise ValueError("warmup must be nonnegative")
        if args.target_per_angle < 2:
            raise ValueError("target-per-angle must be >= 2 (at least 18 pairs overall)")
        if any(not np.isfinite(v) or v <= 0 for v in
               (args.hold_seconds, args.stable_px, args.capture_interval)):
            raise ValueError("Capture timing and stability thresholds must be positive and finite")
        pattern = (args.cols, args.rows)
        print(f"Board: {args.cols}x{args.rows} inner corners, square {args.square_mm:g} mm")
        if not args.offline:
            left, right, size, used, skipped = collect(args, pattern)
        else:
            left, right, size, used, skipped = load_pairs(args.images, pattern)
        report = write_calibration(args, objects, left, right, size, used)
        report['skipped_pairs'] = skipped
        report_path = args.output.with_name(args.output.stem + "_report.json")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Saved: {args.output} and {report_path}")
        if report["stereo_rms_px"] > 1:
            print("WARNING: stereo RMS > 1 px; inspect pairs and recapture before measurement")
        return 0
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        print(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
