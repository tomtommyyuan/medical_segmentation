"""
Tests for the decoding parameter sweep.

A sweep that silently returns the same score for every parameter set looks like
"the defaults are already optimal" and is indistinguishable from a broken
search, so the test that matters is that the grid actually separates.
"""

import itertools

import numpy as np
from synthetic import make_patch

from hover_targets import gen_targets
from tune_postprocess import DEFAULTS, GRID, run_grid


def noisy_cache(n_patches=4, noise=0.12, seed=0):
    """
    Predictions a decent model might make: correct targets plus noise.

    Perfect targets decode perfectly under almost any parameters, which would
    hide the differences the sweep exists to find.
    """
    np_maps, hv_maps, tp_maps, insts, types = [], [], [], [], []

    for k in range(n_patches):
        inst, type_map = make_patch()
        inst = np.ascontiguousarray(np.roll(inst, k * 5, axis=1))
        type_map = np.ascontiguousarray(np.roll(type_map, k * 5, axis=1))

        np_map, hv_map, _ = gen_targets(inst, type_map)
        rng = np.random.default_rng(seed + k)

        np_maps.append(np.clip(np_map + rng.normal(0, noise, np_map.shape), 0, 1).astype(np.float16))
        hv_maps.append(np.clip(hv_map + rng.normal(0, noise * 0.7, hv_map.shape), -1, 1).astype(np.float16))
        tp_maps.append(type_map.astype(np.uint8))
        insts.append(inst.astype(np.int32))
        types.append(type_map.astype(np.uint8))

    return {
        "np": np.stack(np_maps),
        "hv": np.stack(hv_maps),
        "tp": np.stack(tp_maps),
        "true_inst": np.stack(insts),
        "true_type": np.stack(types),
    }


def test_grid_covers_the_current_defaults():
    # Without this the sweep could report a "gain" against a baseline it never
    # actually measured.
    keys = list(GRID)
    combinations = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]

    assert any(all(c[k] == DEFAULTS[k] for k in keys) for c in combinations)


def test_grid_separates_good_parameters_from_bad():
    cache = noisy_cache()
    keys = list(GRID)
    combinations = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]

    rows = run_grid(cache, combinations, n_jobs=1)
    rows.sort(key=lambda r: -r["bpq"])

    assert len(rows) == len(combinations)
    assert rows[0]["bpq"] - rows[-1]["bpq"] > 0.01, "grid does not discriminate"


def test_tuning_beats_the_defaults_on_noisy_predictions():
    cache = noisy_cache()
    keys = list(GRID)
    combinations = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]

    rows = run_grid(cache, combinations, n_jobs=1)
    best = max(rows, key=lambda r: r["bpq"])
    baseline = next(r for r in rows if all(r[k] == DEFAULTS[k] for k in keys))

    assert best["bpq"] >= baseline["bpq"]


def test_parallel_and_serial_agree():
    # The pool inherits the cache through fork rather than pickling it, which
    # is the kind of thing that silently gives every worker an empty cache.
    cache = noisy_cache(n_patches=2)
    keys = list(GRID)
    combinations = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())][:4]

    serial = run_grid(cache, combinations, n_jobs=1)
    parallel = run_grid(cache, combinations, n_jobs=2)

    for a, b in zip(serial, parallel):
        assert np.isclose(a["bpq"], b["bpq"], atol=1e-9)
        assert np.isclose(a["mpq"], b["mpq"], atol=1e-9)


def test_scored_row_carries_its_parameters_and_a_valid_score():
    # score_params reads the module-level cache, so go through run_grid, which
    # installs it.
    cache = noisy_cache(n_patches=1)
    rows = run_grid(cache, [dict(DEFAULTS)], n_jobs=1)

    assert all(rows[0][k] == DEFAULTS[k] for k in DEFAULTS)
    assert 0.0 <= rows[0]["bpq"] <= 1.0
