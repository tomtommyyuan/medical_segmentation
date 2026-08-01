#!/bin/bash
# Submit the ablation grid: one component removed per run, three splits each.
#
# Every row differs from the full model by exactly one flag, so each row's gap
# to the full model is that component's contribution. Runs are tagged, and
# evaluate_instance.py --tag <name> scores whichever one you want.
#
# Usage:
#     bash scripts/run_ablations.sh              # submit to Slurm
#     bash scripts/run_ablations.sh --dry-run    # print what would be submitted
#     bash scripts/run_ablations.sh --evaluate   # score whatever has finished

set -euo pipefail

cd "$(dirname "$0")/.."

# tag                     flags
ABLATIONS=(
    "chroma|"
    "chroma_conv|--encoder conv"
    "chroma_noconsist|--consistency none"
    "chroma_selfdistill|--consistency self"
    "chroma_nofilm|--no-tissue-film"
    "chroma_nologitadj|--tau 0"
    "chroma_frozen|--lora-rank 0"
    "chroma_noaug|--no-augment --consistency none"
)

MODE=${1:-submit}

case "${MODE}" in
    --evaluate)
        for entry in "${ABLATIONS[@]}"; do
            tag="${entry%%|*}"
            if ls results/"${tag}"_split*_best.pth >/dev/null 2>&1; then
                echo "=== ${tag} ==="
                python src/evaluate_instance.py --tag "${tag}"
            else
                echo "=== ${tag}: no checkpoints yet, skipping ==="
            fi
        done
        ;;

    --dry-run|submit)
        for entry in "${ABLATIONS[@]}"; do
            tag="${entry%%|*}"
            flags="${entry#*|}"

            # Flags are exported through the environment rather than through
            # --export=VAR=value. sbatch splits that list on commas and does
            # not quote it, so a value containing a space ("--encoder conv")
            # loses everything after the space.
            if [ "${MODE}" = "--dry-run" ]; then
                printf 'TAG=%q EXTRA=%q sbatch --export=ALL --job-name=%q %s\n' \
                    "${tag}" "${flags}" "${tag}" "scripts/train_chroma.sbatch"
            else
                echo "submitting ${tag}"
                TAG="${tag}" EXTRA="${flags}" \
                    sbatch --export=ALL --job-name="${tag}" scripts/train_chroma.sbatch
            fi
        done
        ;;

    *)
        echo "usage: $0 [--dry-run|--evaluate]" >&2
        exit 1
        ;;
esac
