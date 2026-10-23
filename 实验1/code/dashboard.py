"""Temporal filtering of four-dot measurements and a live chart window (OpenCV + Pillow)."""
from __future__ import annotations

import time
from collections import deque
from functools import lru_cache

import cv2
import numpy as np

EDGE_KEYS = ("top", "right", "bottom", "left")
EDGE_ZH = ("上边 TL-TR", "右边 TR-BR", "下边 BR-BL", "左边 BL-TL")
POINT_LABELS = ("TL", "TR", "BR", "BL")
# BGR, matplotlib tab10 order, so the four edges keep the same colour in every panel.
EDGE_COLORS = ((180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214))
BAR_COLOR = (121, 78, 31)
LABEL_COLOR = (43, 57, 192)
GRID = (225, 225, 225)
AXIS = (40, 40, 40)


@lru_cache(maxsize=16)
def chinese_font(path: str | None = None, size: int = 18):
    from PIL import ImageFont
    candidates = [path] if path else [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    raise ValueError("未找到中文字体。请安装 fonts-noto-cjk，或使用 --font 指定中文字体文件")


def render_panel_text(panel, labels, font):
    """labels: (text, x, y, bgr) drawn at left baseline with one font."""
    return render_text(panel, [(m, x, y, c, None, "ls") for m, x, y, c in labels], font=font)


def render_text(image, labels, font_path: str | None = None, font=None):
    """labels: (text, x, y, bgr, size, Pillow anchor). Pillow is used because putText lacks CJK."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for message, x, y, color, size, anchor in labels:
        f = font if size is None else chinese_font(font_path, size)
        draw.text((x, y), message, font=f, fill=tuple(reversed(color)), anchor=anchor)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _trimmed_mean(samples: np.ndarray, trim: float = .25) -> np.ndarray:
    n = len(samples)
    cut = int(n * trim) if n >= 4 else 0
    return np.sort(samples, axis=0)[cut:n - cut].mean(axis=0)


def _geometry(xyz: np.ndarray) -> dict:
    center = xyz.mean(axis=0)
    normal = np.linalg.svd(xyz - center)[2][-1]
    edges = np.linalg.norm(xyz - np.roll(xyz, -1, axis=0), axis=1)
    return {"xyz_mm": xyz.tolist(), "edges_mm": dict(zip(EDGE_KEYS, edges.tolist())),
            "plane_rms_mm": float(np.sqrt(np.mean(((xyz - center) @ normal)**2))),
            "plane_center_mm": center.tolist(), "plane_normal": normal.tolist()}


class ResultFilter:
    """Sliding-window trimmed mean of the four 3-D centres, with jump/outlier handling.

    Filtering the 3-D centres (not the edge lengths) keeps the four edges geometrically
    consistent. A sample farther than ``jump_mm`` from the window is treated as an outlier;
    ``confirm`` consecutive mutually consistent jumps mean the target really moved, so the
    window restarts there instead of slowly dragging. After ``hold`` invalid frames the
    filter resets; until then it reports the last value marked as stale.
    """

    def __init__(self, window: int = 15, jump_mm: float = 5., confirm: int = 3, hold: int = 15):
        if window < 1 or jump_mm <= 0 or confirm < 1 or hold < 0:
            raise ValueError("Invalid filter parameters")
        self.window, self.jump_mm, self.confirm, self.hold = window, jump_mm, confirm, hold
        self.history: deque = deque(maxlen=window)
        self.pending: list = []
        self.missed = 0
        self.rejected = 0
        self.last: dict | None = None

    def reset(self):
        self.history.clear()
        self.pending.clear()
        self.last = None

    def update(self, result: dict | None) -> dict | None:
        if result is None:
            self.missed += 1
            if self.missed > self.hold:
                self.reset()
            return {**self.last, "stale": True} if self.last else None
        self.missed = 0
        xyz = np.asarray(result["xyz_mm"], dtype=float)
        if self.history:
            reference = _trimmed_mean(np.array(self.history))
            if np.linalg.norm(xyz - reference, axis=1).max() > self.jump_mm:
                self.rejected += 1
                if self.pending and np.linalg.norm(xyz - self.pending[0], axis=1).max() > self.jump_mm:
                    self.pending.clear()
                self.pending.append(xyz)
                if len(self.pending) < self.confirm:
                    return {**self.last, "stale": False, "outlier": True}
                self.history.clear()
                self.history.extend(self.pending)
                self.pending.clear()
            else:
                self.pending.clear()
                self.history.append(xyz)
        else:
            self.history.append(xyz)
        samples = np.array(self.history)
        per_frame = np.linalg.norm(samples - np.roll(samples, -1, axis=1), axis=2)
        self.last = {**_geometry(_trimmed_mean(samples)),
                     "edges_std_mm": dict(zip(EDGE_KEYS, per_frame.std(axis=0).tolist())),
                     "samples": len(samples), "window": self.window,
                     "rejected_total": self.rejected,
                     "reprojection_rms_px": result.get("reprojection_rms_px")}
        return {**self.last, "stale": False, "outlier": False}


def _nice_step(span: float, ticks: int = 6) -> float:
    raw = max(span, 1e-9) / ticks
    magnitude = 10 ** np.floor(np.log10(raw))
    return float(next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw))


def _fmt(value: float, step: float) -> str:
    value = 0. if abs(value) < step * 1e-6 else value  # avoid "-0"
    return f"{value:.0f}" if step >= 1 else f"{value:.{max(0, int(-np.floor(np.log10(step))))}f}"


class Dashboard:
    """Live charts in the style of the lab handout (figures 7 and 8) plus raw-vs-filtered history."""

    def __init__(self, seconds: float = 10., font_path: str | None = None,
                 size: tuple[int, int] = (1280, 800)):
        chinese_font(font_path, 16)  # fail early with a clear message
        self.font_path = font_path
        self.width, self.height = size
        self.seconds = seconds
        self.times: deque = deque()
        self.raw: deque = deque()
        self.filtered: deque = deque()
        self.frame_times: deque = deque(maxlen=30)

    def update(self, raw: dict | None, filtered: dict | None, now: float | None = None):
        now = time.monotonic() if now is None else now
        nan = [np.nan] * 4
        self.times.append(now)
        self.raw.append([raw["edges_mm"][k] for k in EDGE_KEYS] if raw else nan)
        self.filtered.append([filtered["edges_mm"][k] for k in EDGE_KEYS]
                             if filtered and not filtered.get("stale") else nan)
        self.frame_times.append(now)
        while self.times and now - self.times[0] > self.seconds:
            self.times.popleft(); self.raw.popleft(); self.filtered.popleft()

    @property
    def fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.
        return (len(self.frame_times) - 1) / max(self.frame_times[-1] - self.frame_times[0], 1e-9)

    def render(self, raw: dict | None, filtered: dict | None) -> np.ndarray:
        image = np.full((self.height, self.width, 3), 255, np.uint8)
        texts: list = []
        half = self.width // 2
        top_h = self.height // 2
        self._bars(image, texts, (0, 0, half, top_h), raw, filtered)
        self._xy(image, texts, (half, 0, self.width, top_h), filtered)
        self._history(image, texts, (0, top_h, round(self.width * .66), self.height))
        self._table(image, texts, (round(self.width * .66), top_h, self.width, self.height), raw, filtered)
        return render_text(image, texts, self.font_path)

    @staticmethod
    def _text(texts, message, x, y, color=AXIS, size=14, anchor="mm"):
        texts.append((message, float(x), float(y), color, size, anchor))

    def _bars(self, image, texts, rect, raw, filtered):
        x0, y0, x1, y1 = rect
        self._text(texts, "最终结果：四条边长（滤波后）", (x0 + x1) / 2, y0 + 20, size=18)
        left, right, top, bottom = x0 + 60, x1 - 20, y0 + 45, y1 - 40
        values = [filtered["edges_mm"][k] for k in EDGE_KEYS] if filtered else [0.] * 4
        ymax = max(max(values) * 1.15, 10.)
        step = _nice_step(ymax)
        ymax = np.ceil(ymax / step) * step
        scale = (bottom - top) / ymax
        for tick in np.arange(0, ymax + step / 2, step):
            y = round(bottom - tick * scale)
            cv2.line(image, (left, y), (right, y), GRID, 1)
            self._text(texts, _fmt(tick, step), left - 8, y, size=12, anchor="rm")
        cv2.rectangle(image, (left, top), (right, bottom), AXIS, 1)
        self._text(texts, "边长（mm）", left, top - 8, size=13, anchor="ls")
        slot = (right - left) / 4
        stale = not filtered or filtered.get("stale")
        for i, key in enumerate(EDGE_KEYS):
            cx = left + slot * (i + .5)
            bw = slot * .55
            self._text(texts, EDGE_ZH[i], cx, bottom + 18, size=13)
            if not filtered:
                continue
            v = filtered["edges_mm"][key]
            color = (190, 190, 190) if stale else BAR_COLOR
            cv2.rectangle(image, (round(cx - bw / 2), round(bottom - v * scale)),
                          (round(cx + bw / 2), bottom), color, -1)
            sd = filtered["edges_std_mm"][key]
            for y in (bottom - (v - sd) * scale, bottom - (v + sd) * scale):
                cv2.line(image, (round(cx - 6), round(y)), (round(cx + 6), round(y)), AXIS, 1)
            cv2.line(image, (round(cx), round(bottom - (v - sd) * scale)),
                     (round(cx), round(bottom - (v + sd) * scale)), AXIS, 1)
            if raw:
                ry = round(bottom - raw["edges_mm"][key] * scale)
                cv2.line(image, (round(cx - bw / 2), ry), (round(cx + bw / 2), ry), LABEL_COLOR, 2)
            self._text(texts, f"{v:.2f}", cx, bottom - v * scale - 14, size=15)
        if not filtered:
            self._text(texts, "等待有效测量……", (left + right) / 2, (top + bottom) / 2, LABEL_COLOR, 18)
        else:
            self._text(texts, "柱：滤波值  红线：当前帧原始值  误差线：±1σ（窗口内）",
                       right - 4, top + 12, (110, 110, 110), 11, "rm")

    def _xy(self, image, texts, rect, filtered):
        x0, y0, x1, y1 = rect
        self._text(texts, "四个圆心及四条边长（投影到 XY）", (x0 + x1) / 2, y0 + 20, size=18)
        left, right, top, bottom = x0 + 60, x1 - 25, y0 + 45, y1 - 40
        cv2.rectangle(image, (left, top), (right, bottom), AXIS, 1)
        self._text(texts, "X（mm）", (left + right) / 2, bottom + 28, size=13)
        self._text(texts, "Y（mm，向下）", left, top - 8, size=13, anchor="ls")
        if not filtered:
            return
        xy = np.asarray(filtered["xyz_mm"])[:, :2]
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        span = max(float((hi - lo).max()), 1.) * 1.3
        mid = (lo + hi) / 2
        scale = min((right - left), (bottom - top)) / span
        # Equal aspect; Y grows downward so the plot matches the camera image orientation.
        origin = np.array([(left + right) / 2, (top + bottom) / 2])
        to_px = lambda p: origin + (np.asarray(p) - mid) * scale
        step = _nice_step(span / 1.3, 5)
        for axis in (0, 1):
            vmin = mid[axis] - ((right - left) if axis == 0 else (bottom - top)) / 2 / scale
            vmax = 2 * mid[axis] - vmin
            for tick in np.arange(np.ceil(vmin / step) * step, vmax, step):
                p = to_px([tick, mid[1]] if axis == 0 else [mid[0], tick])
                if axis == 0:
                    cv2.line(image, (round(p[0]), top + 1), (round(p[0]), bottom - 1), GRID, 1)
                    self._text(texts, _fmt(tick, step), p[0], bottom + 12, size=11)
                else:
                    cv2.line(image, (left + 1, round(p[1])), (right - 1, round(p[1])), GRID, 1)
                    self._text(texts, _fmt(tick, step), left - 6, p[1], size=11, anchor="rm")
        pts = np.array([to_px(p) for p in xy])
        color = (190, 190, 190) if filtered.get("stale") else BAR_COLOR
        cv2.polylines(image, [np.rint(pts).astype(np.int32)], True, color, 2, cv2.LINE_AA)
        center = pts.mean(axis=0)
        for i, key in enumerate(EDGE_KEYS):
            a, b = pts[i], pts[(i + 1) % 4]
            m = (a + b) / 2
            out = m - center
            m = m + out / max(np.linalg.norm(out), 1e-9) * 16
            self._text(texts, f"{filtered['edges_mm'][key]:.2f} mm", m[0], m[1], LABEL_COLOR, 14)
        for label, p in zip(POINT_LABELS, pts):
            cv2.circle(image, tuple(np.rint(p).astype(int)), 7, color, -1, cv2.LINE_AA)
            out = p - center
            q = p + out / max(np.linalg.norm(out), 1e-9) * 18
            self._text(texts, label, q[0], q[1], BAR_COLOR, 15)

    def _history(self, image, texts, rect):
        x0, y0, x1, y1 = rect
        self._text(texts, f"实时边长（最近 {self.seconds:.0f} 秒）：灰点 原始 / 彩线 滤波 / 红叉 超出范围的原始值",
                   (x0 + x1) / 2, y0 + 16, size=16)
        left, right = x0 + 150, x1 - 15
        top = y0 + 34
        row = (y1 - 10 - top) / 4
        times = np.array(self.times)
        raw = np.array(self.raw).reshape(-1, 4)
        filt = np.array(self.filtered).reshape(-1, 4)
        for i in range(4):
            a, b = round(top + i * row), round(top + (i + 1) * row - 6)
            cv2.rectangle(image, (left, a), (right, b), (200, 200, 200), 1)
            # Scale to the filtered trace; raw outliers (e.g. a wrong 4-dot match) would otherwise
            # flatten the plot. Raw samples far outside are clipped to the border in red.
            base = filt[:, i][np.isfinite(filt[:, i])]
            if not len(base):
                base = raw[:, i][np.isfinite(raw[:, i])]
            values = base
            if len(base):
                near = raw[:, i][np.abs(raw[:, i] - np.median(base)) <= max(np.ptp(base), 0) + 2.]
                values = np.concatenate((base, near))
            current = filt[-1, i] if len(filt) else np.nan
            self._text(texts, EDGE_ZH[i], x0 + 12, (a + b) / 2 - 10, EDGE_COLORS[i], 14, "lm")
            self._text(texts, f"{current:.2f} mm" if np.isfinite(current) else "-- mm",
                       x0 + 12, (a + b) / 2 + 10, AXIS, 14, "lm")
            if not len(values):
                continue
            lo, hi = float(values.min()), float(values.max())
            pad = max((hi - lo) * .15, .05)
            lo, hi = lo - pad, hi + pad
            self._text(texts, f"{hi:.2f}", left - 4, a + 6, (120, 120, 120), 10, "rm")
            self._text(texts, f"{lo:.2f}", left - 4, b - 6, (120, 120, 120), 10, "rm")
            px = right - (times[-1] - times) / self.seconds * (right - left)
            ry = lambda v: b - (v - lo) / (hi - lo) * (b - a)
            for x, v in zip(px, raw[:, i]):
                if not np.isfinite(v):
                    continue
                if lo <= v <= hi:
                    cv2.circle(image, (round(x), round(ry(v))), 1, (160, 160, 160), -1)
                else:
                    cv2.drawMarker(image, (round(x), a + 3 if v > hi else b - 3), LABEL_COLOR,
                                   cv2.MARKER_TILTED_CROSS, 6, 1)
            ok = np.isfinite(filt[:, i])
            # Break the line at gaps (invalid / stale frames).
            for segment in np.split(np.arange(len(ok)), np.where(np.diff(ok.astype(int)) != 0)[0] + 1):
                if len(segment) > 1 and ok[segment[0]]:
                    line = np.column_stack((px[segment], ry(filt[segment, i])))
                    cv2.polylines(image, [np.rint(line).astype(np.int32)], False, EDGE_COLORS[i], 2, cv2.LINE_AA)

    def _table(self, image, texts, rect, raw, filtered):
        x0, y0, x1, y1 = rect
        self._text(texts, "三维圆心（滤波后，左 IR 坐标）", (x0 + x1) / 2, y0 + 16, size=16)
        cols = [x0 + 30, x0 + 110, x0 + 210, x0 + 320]
        y = y0 + 50
        for c, h in zip(cols, ("点", "X (mm)", "Y (mm)", "Z (mm)")):
            self._text(texts, h, c, y, size=13)
        cv2.line(image, (x0 + 10, y + 13), (x1 - 15, y + 13), AXIS, 1)
        for i, label in enumerate(POINT_LABELS):
            y += 28
            self._text(texts, label, cols[0], y, BAR_COLOR, 14)
            if filtered:
                for c, v in zip(cols[1:], filtered["xyz_mm"][i]):
                    self._text(texts, f"{v:.2f}", c, y, size=14)
        cv2.line(image, (x0 + 10, y + 15), (x1 - 15, y + 15), AXIS, 1)
        y += 45
        if filtered:
            state = "保持上次结果（当前帧无效）" if filtered.get("stale") else (
                "当前帧被判为离群，已忽略" if filtered.get("outlier") else "正常")
            lines = [f"状态：{state}",
                     f"滤波窗口：{filtered['samples']}/{filtered['window']} 帧（截尾均值）",
                     f"累计离群帧：{filtered['rejected_total']}",
                     f"平面 RMS：{filtered['plane_rms_mm']:.3f} mm"]
            if raw:
                lines.append(f"重投影 RMS：{raw['reprojection_rms_px']:.3f} px")
                if raw.get("polarity"):
                    lines.append("靶标：" + ("白底黑圆" if raw["polarity"] == "dark" else "黑底白圆"))
        else:
            lines = ["状态：无有效匹配"]
        lines.append(f"刷新率：{self.fps:.1f} fps")
        for line in lines:
            self._text(texts, line, x0 + 20, y, AXIS, 14, "lm")
            y += 26
