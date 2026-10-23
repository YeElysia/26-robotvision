import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo_measure import Calibration, Dot, detect_dots, measure, triangulate


def scene():
    k = np.array([[650., 0, 640], [0, 650., 360], [0, 0, 1]])
    # Unequal principal points exercise the full calibrated triangulation path.
    rk = k.copy()
    rk[0, 2] += 3
    cal = Calibration(k, rk, np.zeros(5), np.zeros(5), np.eye(3),
                      np.array([-.05, 0, 0]), 1280, 720)
    xyz = np.array([[-.03, -.02, .7], [.03, -.02, .7],
                    [.03, .02, .7], [-.03, .02, .7]])
    left, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, cal.left_dist)
    right, _ = cv2.projectPoints(xyz, np.zeros(3), cal.translation_m, rk, cal.right_dist)
    return cal, xyz, left.reshape(-1, 2), right.reshape(-1, 2)


def test_triangulation_and_matching():
    cal, xyz, left, right = scene()
    reconstructed, error = triangulate(left, right, cal)
    np.testing.assert_allclose(reconstructed, xyz * 1000, atol=1e-7)
    assert error < 1e-8
    ld = [Dot(p, 5, .9) for p in left]
    rd = [Dot(p, 5, .9) for p in right[[2, 0, 3, 1]]]
    result = measure(ld, rd, cal)
    assert result is not None
    np.testing.assert_allclose(list(result['edges_mm'].values()), [60, 40, 60, 40], atol=1e-7)
    assert result['plane_rms_mm'] < 1e-8


def test_bad_matches_rejected():
    cal, _, left, right = scene()
    ld = [Dot(p, 5, .9) for p in left]
    rd = [Dot(p + [0, 15], 5, .9) for p in right]
    assert measure(ld, rd, cal) is None
    assert measure([], rd, cal) is None
    # Swap cameras: negative depth must not be accepted.
    assert measure([Dot(p, 5, .9) for p in right], [Dot(p, 5, .9) for p in left], cal) is None


def test_detection_and_offline_cli(tmp_path):
    cal, _, left, right = scene()
    for name, points in [('left', left), ('right', right)]:
        image = np.full((720, 1280), 240, dtype=np.uint8)
        for point in points:
            cv2.circle(image, tuple(np.rint(point).astype(int)), 7, 10, -1)
        dots, _ = detect_dots(image, 15, 800, .65)
        assert len(dots) == 4
        cv2.imwrite(str(tmp_path / f'{name}.png'), image)
    cal.save(tmp_path / 'calibration.json')
    loaded = Calibration.load(tmp_path / 'calibration.json')
    np.testing.assert_array_equal(loaded.right_k, cal.right_k)
    output = tmp_path / 'output'
    run = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'stereo_measure.py'),
                          '--offline', str(tmp_path), '--headless', '--output', str(output)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    payload = json.loads(next(output.glob('*/result.json')).read_text())
    assert payload['valid']
    # Rasterized circles have integer centers; compare with their actual quantized geometry,
    # not ideal subpixel coordinates (a single disparity pixel changes recovered depth).
    quantized_xyz, _ = triangulate(np.rint(left), np.rint(right), cal)
    expected = np.linalg.norm(quantized_xyz - np.roll(quantized_xyz, -1, axis=0), axis=1)
    np.testing.assert_allclose(list(payload['measurement']['edges_mm'].values()), expected, atol=1e-6)
    assert next(output.glob('*/edges.csv')).is_file()


def test_live_line_formats():
    from stereo_measure import live_line, parser
    result = {"edges_mm": {"top": 100., "right": 80., "bottom": 100.5, "left": 79.5},
              "plane_center_mm": [0., 0., 450.], "plane_rms_mm": .1, "reprojection_rms_px": .2}
    text = live_line(result, [1] * 4, [1] * 4, "text")
    assert "top  100.00" in text and "Z 450.0 mm" in text
    record = json.loads(live_line(result, [1] * 4, [1] * 4, "jsonl"))
    assert record["valid"] and record["edges_mm"]["bottom"] == 100.5
    assert json.loads(live_line(None, [1] * 5, [], "jsonl")) == {**json.loads(
        live_line(None, [1] * 5, [], "jsonl")), "valid": False, "candidates": [5, 0]}
    assert "无有效匹配" in live_line(None, [], [], "text")
    assert parser().parse_args(["--live"]).live == "text"
    assert parser().parse_args(["--live", "jsonl"]).live == "jsonl"


def test_auto_polarity_handles_both_targets():
    from types import SimpleNamespace
    from stereo_measure import detect_and_measure
    cal, _, left, right = scene()
    args = SimpleNamespace(min_area=15, max_area=800, circularity=.65, roi=None, epipolar_px=2,
                           reprojection_px=1.5, min_z_mm=150, max_z_mm=3000)

    def render(points, paper, ink):
        image = np.full((720, 1280), 128, np.uint8)
        cv2.rectangle(image, (400, 200), (880, 520), paper, -1)
        for point in points:
            cv2.circle(image, tuple(np.rint(point).astype(int)), 7, ink, -1)
        return image

    for paper, ink, expected in ((240, 10, "dark"), (15, 245, "light")):
        l, r = render(left, paper, ink), render(right, paper, ink)
        for order in (("dark", "light"), ("light", "dark")):
            result, *_, polarity = detect_and_measure(l, r, cal, args, order)
            assert result is not None and polarity == expected == result["polarity"]
        wrong = "light" if expected == "dark" else "dark"
        assert detect_and_measure(l, r, cal, args, (wrong,))[0] is None
