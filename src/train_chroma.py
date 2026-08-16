"""
Train CHROMA-Net on one split of the official PanNuke three-fold protocol.

Trains the multi-task objective (nuclei, distance maps, class) and optionally
the stain consistency term, selecting the best checkpoint by validation mPQ
rather than by loss, since mPQ is what the benchmark reports and the two do not
move together.

Consistency modes:
    none      supervised only, one forward pass
    self      student's strong view matched to its own detached weak view
    teacher   student's strong view matched to an EMA teacher's weak view

The two views of a patch differ only in stain, and stain augmentation never
moves a pixel, so they are exactly registered and no warping is needed. The
consistency weight ramps in: the teacher starts as a copy of a randomly
initialised student, and its targets are noise until it has learned something.

Run all three splits to get a comparable number:
    python src/train_chroma.py --split 1
    python src/train_chroma.py --split 2
    python src/train_chroma.py --split 3
    python src/evaluate_instance.py --model chroma

Usage:
    python src/train_chroma.py --split 1 --encoder phikon
    python src/train_chroma.py --split 1 --encoder conv --consistency none
    python src/train_chroma.py --split 1 --limit-batches 4 --epochs 1   # smoke test
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from chroma_net import ChromaNet
from dataset import SPLITS, TISSUE_TYPES, NucleiDataset
from losses import ChromaLoss, compute_class_prior, consistency_loss, sigmoid_rampup
from postprocess import decode_batch
from pq_metrics import aggregate_pq, pq_per_image

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

EPOCHS = 50
LR = 3e-4
BATCH_SIZE = 16
NUM_WORKERS = 8
WARMUP_EPOCHS = 2
EMA_DECAY = 0.999
CONSISTENCY_WEIGHT = 1.0
RAMPUP_EPOCHS = 5
# Patches scored with the full instance pipeline each epoch. Decoding and PQ
# run on CPU at roughly 100ms a patch, so scoring a whole fold every epoch
# costs more than the epoch itself. evaluate_instance.py scores everything.
VAL_SUBSET = 512


def set_seed(seed):
    """Seed every generator that affects a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_schedule(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay, stepped per optimizer step."""

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMATeacher:
    """
    Exponential moving average of the student, used as the consistency target.

    The decay ramps in from the start. A fixed 0.999 would leave the teacher
    essentially at the random initialisation for the first thousand steps,
    which is not a target worth matching.
    """

    def __init__(self, model, decay=EMA_DECAY):
        self.model = copy.deepcopy(model)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, student, step):
        decay = min(self.decay, (1.0 + step) / (10.0 + step))
        student_state = student.state_dict()

        for key, value in self.model.state_dict().items():
            source = student_state[key]
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source.detach(), alpha=1.0 - decay)
            else:
                # Buffers such as num_batches_tracked are integer counters.
                value.copy_(source)

    def to(self, device):
        self.model.to(device)
        return self


def build_model(args, device):
    model = ChromaNet(
        encoder=args.encoder,
        lora_rank=args.lora_rank,
        tissue_film=not args.no_tissue_film,
        pretrained=not args.no_pretrained,
    )
    return model.to(device)


def autocast_context(args, device):
    if args.amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def train_one_epoch(model, teacher, loader, criterion, optimizer, scheduler,
                    device, args, global_step):
    model.train()

    totals = {"loss": 0.0, "consistency": 0.0}
    n_samples = 0

    rampup_steps = args.rampup_epochs * len(loader)

    for batch_index, batch in enumerate(loader):
        if args.limit_batches and batch_index >= args.limit_batches:
            break

        image = batch["image"].to(device, non_blocking=True)
        tissue = batch["tissue"].to(device, non_blocking=True)
        targets = {
            "np_map": batch["np_map"].to(device, non_blocking=True),
            "hv_map": batch["hv_map"].to(device, non_blocking=True),
            "tp_map": batch["tp_map"].to(device, non_blocking=True),
        }

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(args, device):
            outputs = model(image, tissue)
            loss, _ = criterion(outputs, targets)

            consistency = torch.zeros((), device=device)
            if args.consistency != "none":
                strong = batch["image_strong"].to(device, non_blocking=True)
                student_strong = model(strong, tissue)

                if args.consistency == "teacher":
                    with torch.no_grad():
                        target_outputs = teacher.model(image, tissue)
                else:
                    target_outputs = {k: v.detach() for k, v in outputs.items()}

                weight = args.consistency_weight * sigmoid_rampup(global_step, rampup_steps)
                consistency = weight * consistency_loss(student_strong, target_outputs)
                loss = loss + consistency

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        scheduler.step()

        global_step += 1
        if teacher is not None:
            teacher.update(model, global_step)

        batch_size = image.size(0)
        totals["loss"] += float(loss.detach()) * batch_size
        totals["consistency"] += float(consistency.detach()) * batch_size
        n_samples += batch_size

    n_samples = max(n_samples, 1)
    return totals["loss"] / n_samples, totals["consistency"] / n_samples, global_step


@torch.no_grad()
def validate(model, loader, criterion, device, args, max_patches=VAL_SUBSET):
    """
    Validation loss plus the instance metrics the benchmark reports.

    Decoding and PQ are CPU-bound, so only the first max_patches patches are
    scored. The ranking between epochs is stable enough for checkpoint
    selection; evaluate_instance.py scores the full fold. Pass 0 to score
    everything.
    """
    model.eval()

    if max_patches <= 0:
        max_patches = len(loader.dataset)

    total_loss = 0.0
    n_samples = 0
    bpq_list = []
    class_pq_list = []
    tissues = []
    scored = 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        tissue = batch["tissue"].to(device, non_blocking=True)
        targets = {
            "np_map": batch["np_map"].to(device, non_blocking=True),
            "hv_map": batch["hv_map"].to(device, non_blocking=True),
            "tp_map": batch["tp_map"].to(device, non_blocking=True),
        }

        with autocast_context(args, device):
            outputs = model(image, tissue)
            loss, _ = criterion(outputs, targets)

        total_loss += float(loss) * image.size(0)
        n_samples += image.size(0)

        if scored >= max_patches:
            continue

        # Reuse the forward pass already made for the loss.
        np_prob = torch.sigmoid(outputs["np"]).float().squeeze(1).cpu().numpy()
        hv_pred = outputs["hv"].float().cpu().numpy()
        tp_pred = outputs["tp"].float().argmax(dim=1).cpu().numpy().astype(np.uint8)

        pred_inst, pred_type = decode_batch(np_prob, hv_pred, tp_pred)

        true_inst = batch["inst"].numpy()
        true_type = batch["type"].numpy()

        for i in range(len(pred_inst)):
            if scored >= max_patches:
                break
            bpq, class_pq = pq_per_image(true_inst[i], true_type[i], pred_inst[i], pred_type[i])
            bpq_list.append(bpq)
            class_pq_list.append(class_pq)
            tissues.append(TISSUE_TYPES[int(batch["tissue"][i])])
            scored += 1

    results = aggregate_pq(bpq_list, class_pq_list, tissues) if bpq_list else None

    return {
        "loss": total_loss / max(n_samples, 1),
        "mpq": results["mpq"] if results else float("nan"),
        "bpq": results["bpq"] if results else float("nan"),
        "n_scored": scored,
    }


def main():
    parser = argparse.ArgumentParser(description="Train CHROMA-Net on a PanNuke split")
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--encoder", type=str, default="phikon",
                        help="phikon, uni, conv, or a timm/hub model name")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="0 freezes the encoder entirely")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="logit adjustment strength, 0 disables it")
    parser.add_argument("--no-tissue-film", action="store_true",
                        help="drop tissue conditioning, for the unconditioned number")
    parser.add_argument("--consistency", type=str, default="teacher",
                        choices=["none", "self", "teacher"])
    parser.add_argument("--consistency-weight", type=float, default=CONSISTENCY_WEIGHT)
    parser.add_argument("--rampup-epochs", type=int, default=RAMPUP_EPOCHS)
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--val-subset", type=int, default=VAL_SUBSET,
                        help="patches scored with the full instance pipeline per epoch, 0 for all")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="random encoder init, for debugging without a download")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="chroma")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--out-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--save-every", type=int, default=0,
                        help="also keep every Nth epoch's checkpoint; 0 keeps only the best")
    parser.add_argument("--limit-batches", type=int, default=0,
                        help="stop each epoch after N batches, for smoke tests")
    args = parser.parse_args()

    set_seed(args.seed)

    run = f"{args.tag}_split{args.split}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds = SPLITS[args.split]
    print(f"[{run}] Device: {device}")
    print(f"[{run}] Folds: train={folds['train']} val={folds['val']} test={folds['test']}")
    print(f"[{run}] LR: {args.lr}, Batch: {args.batch_size}, Epochs: {args.epochs}")
    print(f"[{run}] Encoder: {args.encoder}, LoRA rank: {args.lora_rank}, "
          f"consistency: {args.consistency}, tau: {args.tau}")

    train_dataset = NucleiDataset(
        folds["train"], data_dir=args.data_dir,
        augment=not args.no_augment,
        stain_views=args.consistency != "none",
        seed=args.seed,
    )
    val_dataset = NucleiDataset(
        folds["val"], data_dir=args.data_dir, return_instances=True, seed=args.seed,
    )
    print(f"[{run}] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)

    model = build_model(args, device)
    trainable, total = model.parameter_summary()
    print(f"[{run}] Parameters: {trainable:,} trainable / {total:,} total "
          f"({100 * trainable / total:.1f}%), LoRA layers: {model.lora_layers}")

    class_prior = None
    if args.tau > 0:
        class_prior = compute_class_prior(train_dataset.data["types"])
        print(f"[{run}] Class prior: {np.round(class_prior, 5).tolist()}")

    criterion = ChromaLoss(class_prior=class_prior, tau=args.tau).to(device)

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    steps_per_epoch = args.limit_batches or len(train_loader)
    scheduler = cosine_schedule(optimizer, args.warmup_epochs * steps_per_epoch,
                                args.epochs * steps_per_epoch)

    teacher = None
    if args.consistency == "teacher":
        teacher = EMATeacher(model, decay=args.ema_decay).to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, f"{run}_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, f"{run}_config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)

    csv_path = os.path.join(args.out_dir, f"{run}_history.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "consistency", "val_loss",
                         "val_mpq", "val_bpq", "lr", "time_s"])

    best_mpq = -1.0
    best_epoch = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, consistency, global_step = train_one_epoch(
            model, teacher, train_loader, criterion, optimizer, scheduler,
            device, args, global_step,
        )
        metrics = validate(model, val_loader, criterion, device, args, args.val_subset)

        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"[{run}] Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Cons: {consistency:.4f} | "
              f"Val Loss: {metrics['loss']:.4f} | "
              f"Val mPQ: {metrics['mpq']:.4f} | "
              f"Val bPQ: {metrics['bpq']:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {elapsed:.1f}s")

        csv_writer.writerow([epoch, f"{train_loss:.6f}", f"{consistency:.6f}",
                             f"{metrics['loss']:.6f}", f"{metrics['mpq']:.6f}",
                             f"{metrics['bpq']:.6f}", f"{current_lr:.3e}",
                             f"{elapsed:.1f}"])
        csv_file.flush()

        # Selected on mPQ, not loss: the benchmark reports mPQ and the two do
        # not peak at the same epoch. Epoch 1 always writes, so a run whose mPQ
        # is NaN or never rises still leaves a loadable checkpoint and fails
        # later on a bad score rather than on a missing file.
        if epoch == 1 or metrics["mpq"] > best_mpq:
            best_mpq = metrics["mpq"]
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                 "val_mpq": metrics["mpq"], "val_bpq": metrics["bpq"]},
                os.path.join(args.out_dir, f"{run}_best.pth"),
            )

        # Off by default. state_dict() holds the frozen encoder too, so a
        # checkpoint is ~1.3 GB whatever the trainable count says; keeping one
        # per epoch is 63 GB per split, and the ablation grid would be half a
        # terabyte. The best checkpoint above is what evaluation loads.
        if args.save_every and epoch % args.save_every == 0:
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                       os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

    csv_file.close()

    print(f"\n[{run}] Best val mPQ: {best_mpq:.4f} at epoch {best_epoch}")
    print(f"[{run}] Best checkpoint: {os.path.join(args.out_dir, f'{run}_best.pth')}")
    print(f"[{run}] History: {csv_path}")


if __name__ == "__main__":
    main()
