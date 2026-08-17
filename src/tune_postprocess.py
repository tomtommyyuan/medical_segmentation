"""
Grid-search the watershed decoding parameters on validation.

The four parameters in postprocess.py were set by hand and never tuned. They
control merge and split behaviour directly, which is most of what bPQ measures,
so the defaults are as likely to be costing points as not.

Tuning them needs no retraining. Network predictions for a fold are computed
once and cached, then every grid point re-decodes the same cached maps. That is
CPU work only, and each grid point is independent, so it parallelises across
the cores the job already has.

Searched on the VALIDATION fold. Tuning on test and then reporting test is how
a benchmark result becomes meaningless, and the whole point of the protocol
work in this repo is that its numbers mean something.

Usage:
    python src/tune_postprocess.py --tag chroma --split 1
    python src/tune_postprocess.py --tag chroma --split 1 --n-patches 400 --tta
    python src/tune_postprocess.py --tag chroma --splits 1 2 3   # slower, safer
"""

import argparse
import itertools
import json
import multiprocessing
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SPLITS, NucleiDataset
from evaluate_instance import load_chroma
from postprocess import assign_types, decode_instances
from pq_metrics import pq_per_image
from tta import predict_plain, predict_tta

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Defaults currently in postprocess.py, kept first so the baseline row is
# always in the table.
DEFAULTS = {"np_threshold": 0.5, "marker_threshold": 0.4, "min_size": 10, "sobel_ksize": 21}

GRID = {
    "np_threshold": [0.4, 0.5, 0.6],
    "marker_threshold": [0.3, 0.4, 0.5],
    "min_size": [5, 10, 20],
    "sobel_ksize": [11, 21],
}

# Set before the worker pool forks, so children inherit it copy-on-write
# instead of pickling several hundred megabytes per task.
_CACHE = {}


def build_cache(args, split, device):
    """Run the model once over the validation fold and keep its raw outputs."""
    fold = SPLITS[split]["val"]
    dataset = NucleiDataset(fold, data_dir=args.data_dir, return_instances=True)

    limit = min(args.n_patches, len(dataset)) if args.n_patches else len(dataset)
    print(f"  split {split}: caching {limit} patches from {fold}")

    checkpoint_path = os.path.join(args.out_dir, f"{args.tag}_split{split}_best.pth")
    model, checkpoint = load_chroma(checkpoint_path, device)
    print(f"  checkpoint epoch {checkpoint['epoch']}, val mPQ {checkpoint['val_mpq']:.4f}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    np_probs, hv_preds, tp_preds, true_insts, true_types = [], [], [], [], []
    seen = 0

    for batch in loader:
        if seen >= limit:
            break

        image = batch["image"].to(device, non_blocking=True)
        tissue = batch["tissue"].to(device, non_blocking=True)

        predict = predict_tta if args.tta else predict_plain
        predictions = predict(model, image, tissue)

        # float16 halves the cache; decoding immediately casts back to float32
        # and the thresholds involved are nowhere near that precision.
        np_probs.append(predictions["np"].float().squeeze(1).cpu().numpy().astype(np.float16))
        hv_preds.append(predictions["hv"].float().cpu().numpy().astype(np.float16))
        tp_preds.append(predictions["tp"].float().argmax(dim=1).cpu().numpy().astype(np.uint8))
        true_insts.append(batch["inst"].numpy().astype(np.int32))
        true_types.append(batch["type"].numpy().astype(np.uint8))

        seen += image.shape[0]

    return {
        "np": np.concatenate(np_probs)[:limit],
        "hv": np.concatenate(hv_preds)[:limit],
        "tp": np.concatenate(tp_preds)[:limit],
        "true_inst": np.concatenate(true_insts)[:limit],
        "true_type": np.concatenate(true_types)[:limit],
    }


def score_params(params):
    """Decode the cached predictions with one parameter set and score them."""
    cache = _CACHE
    bpq_total, mpq_total, n = 0.0, 0.0, 0

    for i in range(len(cache["np"])):
        pred_inst = decode_instances(
            cache["np"][i].astype(np.float32),
            cache["hv"][i].astype(np.float32),
            **params,
        )
        pred_type = assign_types(pred_inst, cache["tp"][i])

        bpq, class_pq, _, _ = pq_per_image(
            cache["true_inst"][i], cache["true_type"][i], pred_inst, pred_type
        )

        bpq_total += bpq
        if not np.all(np.isnan(class_pq)):
            mpq_total += np.nanmean(class_pq)
        n += 1

    return {**params, "bpq": bpq_total / max(n, 1), "mpq": mpq_total / max(n, 1)}


def _init_worker(cache):
    global _CACHE
    _CACHE = cache


def run_grid(cache, combinations, n_jobs):
    """Score every parameter combination, in parallel where possible."""
    global _CACHE
    _CACHE = cache

    if n_jobs <= 1:
        return [score_params(p) for p in combinations]

    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        # No fork available; pickling the cache per task would cost more than
        # the parallelism saves.
        return [score_params(p) for p in combinations]

    with context.Pool(n_jobs, initializer=_init_worker, initargs=(cache,)) as pool:
        return pool.map(score_params, combinations)


def main():
    parser = argparse.ArgumentParser(description="Tune watershed decoding on validation")
    parser.add_argument("--tag", type=str, default="chroma")
    parser.add_argument("--splits", type=int, nargs="+", default=[1], choices=[1, 2, 3])
    parser.add_argument("--n-patches", type=int, default=250,
                        help="validation patches to tune on; 0 uses the whole fold")
    parser.add_argument("--tta", action="store_true",
                        help="tune against TTA predictions, if that is how you will report")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--jobs", type=int, default=8, help="parallel grid evaluations")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--out-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    keys = list(GRID)
    combinations = [dict(zip(keys, values)) for values in itertools.product(*GRID.values())]
    # Put the current defaults first so the baseline is unmissable.
    combinations.sort(key=lambda c: c != DEFAULTS)

    print(f"Device: {device}")
    print(f"Tuning {args.tag} on validation folds of splits {args.splits}")
    print(f"{len(combinations)} parameter combinations, {args.jobs} parallel jobs")

    # Averaged over splits so a single fold's quirks do not pick the winner.
    totals = {}
    for split in args.splits:
        cache = build_cache(args, split, device)

        start = time.time()
        rows = run_grid(cache, combinations, args.jobs)
        print(f"  scored in {time.time() - start:.0f}s")

        for row in rows:
            key = tuple(row[k] for k in keys)
            entry = totals.setdefault(key, {"bpq": 0.0, "mpq": 0.0, "n": 0})
            entry["bpq"] += row["bpq"]
            entry["mpq"] += row["mpq"]
            entry["n"] += 1

    results = []
    for key, entry in totals.items():
        results.append({
            **dict(zip(keys, key)),
            "bpq": entry["bpq"] / entry["n"],
            "mpq": entry["mpq"] / entry["n"],
        })

    results.sort(key=lambda r: -r["bpq"])
    baseline = next(r for r in results if all(r[k] == DEFAULTS[k] for k in keys))
    best = results[0]

    print(f"\n{'=' * 78}")
    print("  TOP 10 BY VALIDATION bPQ")
    print(f"{'=' * 78}")
    header = "  " + "".join(f"{k:>18}" for k in keys) + f"{'bPQ':>10}{'mPQ':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in results[:10]:
        line = "  " + "".join(f"{row[k]:>18}" for k in keys)
        print(line + f"{row['bpq']:>10.4f}{row['mpq']:>10.4f}")

    print(f"\n  current defaults: bPQ {baseline['bpq']:.4f}  mPQ {baseline['mpq']:.4f}")
    print(f"  best found:       bPQ {best['bpq']:.4f}  mPQ {best['mpq']:.4f}")
    print(f"  gain:             bPQ {best['bpq'] - baseline['bpq']:+.4f}  "
          f"mPQ {best['mpq'] - baseline['mpq']:+.4f}")

    if best["bpq"] - baseline["bpq"] < 0.002:
        print("\n  Within noise. The defaults are fine and decoding is not the bottleneck;")
        print("  the deficit is in the model, not the post-processing.")

    payload = {
        "tag": args.tag,
        "splits": args.splits,
        "n_patches": args.n_patches,
        "tta": args.tta,
        "best": {k: best[k] for k in keys},
        "best_scores": {"bpq": best["bpq"], "mpq": best["mpq"]},
        "baseline_scores": {"bpq": baseline["bpq"], "mpq": baseline["mpq"]},
        "all": results,
    }
    path = os.path.join(args.out_dir, f"{args.tag}_decode_params.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nSaved to: {path}")
    print("\nApply on the test folds with:")
    flags = " ".join(f"--{k.replace('_', '-')} {best[k]}" for k in keys)
    print(f"    python src/evaluate_instance.py --tag {args.tag} --tta {flags}")


if __name__ == "__main__":
    main()
