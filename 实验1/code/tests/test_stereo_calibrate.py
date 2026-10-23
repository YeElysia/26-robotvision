import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo_calibrate import (Coverage, board_pose, board_points, calibrate_pairs,
                              collect, find_corners, load_pairs, parser, write_calibration)
from stereo_measure import Calibration


def test_board_defaults_and_detection():
    args = parser().parse_args([])
    points = board_points(args.cols, args.rows, args.square_mm / 1000)
    assert points.shape == (88, 3)
    np.testing.assert_allclose(points[-1], [.2, .14, 0], atol=1e-7)
    image = np.full((440, 570), 255, np.uint8)
    for y in range(9):
        for x in range(12):
            if (x + y) % 2 == 0:
                image[40+y*40:40+(y+1)*40, 45+x*40:45+(x+1)*40] = 0
    corners = find_corners(image, (11, 8))
    assert corners is not None and corners.shape == (88, 1, 2)
    assert find_corners(np.full_like(image, 255), (11, 8)) is None
    with pytest.raises(ValueError):
        board_points(11, 8, 0)


def test_synthetic_calibration(tmp_path):
    objects = board_points(11, 8, .02)
    lk = np.array([[800., 0, 640], [0, 810., 360], [0, 0, 1]])
    rk = np.array([[805., 0, 638], [0, 815., 362], [0, 0, 1]])
    rotation = cv2.Rodrigues(np.array([.002, -.004, .001]))[0]
    translation = np.array([-.05, .0002, .0001])
    left, right = [], []
    rng = np.random.default_rng(42)
    for _ in range(18):
        rv = rng.uniform(-.4, .4, 3)
        tv = np.array([rng.uniform(-.17, .02), rng.uniform(-.14, .02), rng.uniform(.5, .9)])
        right_rv = cv2.Rodrigues(rotation @ cv2.Rodrigues(rv)[0])[0]
        left.append(cv2.projectPoints(objects, rv, tv, lk, np.zeros(5))[0])
        right.append(cv2.projectPoints(objects, right_rv, rotation @ tv + translation,
                                       rk, np.zeros(5))[0])
    cal, report = calibrate_pairs(objects, left, right, (1280, 720))
    np.testing.assert_allclose(cal.translation_m, translation, atol=1e-5)
    np.testing.assert_allclose(cal.rotation, rotation, atol=1e-5)
    np.testing.assert_allclose(cal.left_k, lk, atol=.02)
    assert report['stereo_rms_px'] < .001
    assert len(report['left_per_view_rms_px']) == 18
    cal.save(tmp_path / 'calibration.json')
    loaded = Calibration.load(tmp_path / 'calibration.json')
    np.testing.assert_allclose(loaded.translation_m, translation, atol=1e-5)
    with pytest.raises(ValueError, match='At least 10'):
        calibrate_pairs(objects, left[:3], right[:3], (1280, 720))


def test_match_order_reverses_flipped_right_corners():
    from stereo_calibrate import match_order
    left = board_points(11, 8, 1.)[:, None, :2].copy()
    right = left - [5, 0]
    np.testing.assert_array_equal(match_order(left, right), right)
    np.testing.assert_array_equal(match_order(left, right[::-1]), right)


def test_pair_resolution_rejected(tmp_path):
    pair = tmp_path / 'pair'
    pair.mkdir()
    cv2.imwrite(str(pair / 'left.png'), np.zeros((100, 200), np.uint8))
    cv2.imwrite(str(pair / 'right.png'), np.zeros((101, 200), np.uint8))
    with pytest.raises(ValueError, match='same original resolution'):
        load_pairs(tmp_path, (11, 8))


def test_pose_and_coverage():
    objects = board_points(11, 8, .02)
    k = np.array([[800., 0, 640], [0, 800., 360], [0, 0, 1]])
    coverage = Coverage(3)
    for rv, expected in [(np.zeros(3), 4), (np.array([0., .4, 0]), 5),
                         (np.array([.4, 0., 0]), 1)]:
        corners = cv2.projectPoints(objects, rv, np.array([-.1, -.08, .7]), k, np.zeros(5))[0]
        pose = board_pose(objects, corners, k, np.zeros(5))
        assert pose['bin'] == expected
        assert pose['pose_rms_px'] < .001
        assert coverage.novel(pose, (1280, 720))
        coverage.poses.append(pose)
        assert not coverage.novel(pose, (1280, 720))
    assert sum(coverage.counts) == 3
    assert coverage.report()['progress_percent'] == pytest.approx(100 / 9)


def test_chinese_panel_rendering():
    from stereo_calibrate import chinese_font, render_panel_text
    try:
        font = chinese_font()
    except ValueError:
        pytest.skip('No system Chinese font installed')
    panel = np.zeros((50, 300, 3), np.uint8)
    result = render_panel_text(panel, [('自动采集：正面', 5, 30, (0, 255, 0))], font)
    assert result.shape == panel.shape
    assert result[:, :, 1].sum() > 0
    assert result[:, :, 0].sum() == 0


def test_auto_capture_completes_and_saves(tmp_path, monkeypatch):
    import stereo_calibrate as module
    from types import SimpleNamespace

    args = parser().parse_args(['--images', str(tmp_path / 'images'),
                                '--output', str(tmp_path / 'result.json'),
                                '--target-per-angle', '2', '--hold-seconds', '.1',
                                '--capture-interval', '.1'])
    cal = Calibration(np.eye(3), np.eye(3), np.zeros(5), np.zeros(5), np.eye(3),
                      np.array([-.05, 0, 0]), 320, 240)
    closed = []
    source = SimpleNamespace(cal=cal, metadata={}, read=lambda n: (
        np.zeros((240, 320), np.uint8), np.zeros((240, 320), np.uint8), None),
        close=lambda: closed.append(True))
    monkeypatch.setattr(module, 'RealSenseSource', lambda args: source)
    monkeypatch.setattr(module, 'find_corners', lambda *a, **kw: np.zeros((88, 1, 2), np.float32))
    monkeypatch.setattr(module.Coverage, 'novel', lambda *a: True)
    def pose(*a):
        count = len(list(args.images.glob('*/left.png')))
        return dict(bin=count % 9, center_px=[100, 100], yaw_deg=0., pitch_deg=0.,
                    distance_m=.7, pose_rms_px=0.)
    monkeypatch.setattr(module, 'board_pose', pose)
    clock = iter(range(1000))
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock))
    for name in ('namedWindow', 'resizeWindow', 'imshow', 'destroyAllWindows', 'drawChessboardCorners'):
        monkeypatch.setattr(cv2, name, lambda *a: None)
    monkeypatch.setattr(cv2, 'waitKey', lambda *a: -1)
    left, right, size, used, _ = collect(args, (11, 8))
    assert len(used) == len(left) == len(right) == 18
    assert closed == [True]
    assert size == (320, 240)
    import json
    progress = json.loads((args.images / 'capture_progress.json').read_text())
    assert progress['progress_percent'] == 100
    monkeypatch.setattr(module, 'calibrate_pairs', lambda *a: (cal, {'pairs': 18, 'stereo_rms_px': .1}))
    write_calibration(args, board_points(11, 8, .02), left, right, size, used)
    assert args.output.is_file()
    report = json.loads((tmp_path / 'result_report.json').read_text())
    assert report['pairs'] == 18
    assert len(report['used_pairs']) == 18
