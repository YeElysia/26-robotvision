import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import Dashboard, ResultFilter

SQUARE = np.array([[0., 0, 500], [60, 0, 500], [60, 40, 500], [0, 40, 500]])


def sample(xyz):
    return {"xyz_mm": np.asarray(xyz).tolist(), "reprojection_rms_px": .1,
            "edges_mm": dict(zip(("top", "right", "bottom", "left"),
                                 np.linalg.norm(xyz - np.roll(xyz, -1, axis=0), axis=1).tolist()))}


def test_filter_reduces_noise_and_rejects_outlier():
    rng = np.random.default_rng(0)
    f = ResultFilter(window=15, jump_mm=5)
    raw_top, filt_top = [], []
    for _ in range(60):
        xyz = SQUARE + rng.normal(0, .3, SQUARE.shape)
        out = f.update(sample(xyz))
        raw_top.append(np.linalg.norm(xyz[1] - xyz[0]))
        filt_top.append(out["edges_mm"]["top"])
    assert np.std(filt_top[20:]) < np.std(raw_top[20:]) / 2
    assert abs(filt_top[-1] - 60) < .3
    before = f.update(sample(SQUARE))["edges_mm"]["top"]
    out = f.update(sample(SQUARE + [[0, 0, 0], [30, 0, 0], [0, 0, 0], [0, 0, 0]]))
    assert out["outlier"] and out["edges_mm"]["top"] == before
    assert f.rejected == 1


def test_filter_follows_real_move_and_holds_on_gaps():
    f = ResultFilter(window=15, jump_mm=5, confirm=3, hold=2)
    for _ in range(10):
        f.update(sample(SQUARE))
    moved = SQUARE + [20, 0, 0]
    outs = [f.update(sample(moved)) for _ in range(3)]
    assert outs[0]["outlier"] and outs[1]["outlier"]
    np.testing.assert_allclose(outs[2]["xyz_mm"], moved)
    assert f.update(None)["stale"]
    assert f.update(None)["stale"]
    assert f.update(None) is None
    with pytest.raises(ValueError):
        ResultFilter(window=0)


def test_dashboard_renders_with_and_without_data():
    try:
        board = Dashboard(5)
    except ValueError:
        pytest.skip("No system Chinese font installed")
    f = ResultFilter()
    for t in range(20):
        raw = sample(SQUARE + .1 * np.sin(t))
        board.update(raw, f.update(raw), now=t / 30)
    image = board.render(raw, f.last)
    assert image.shape == (800, 1280, 3) and image.std() > 10
    board.update(None, None, now=1.)
    assert board.render(None, None).shape == (800, 1280, 3)
