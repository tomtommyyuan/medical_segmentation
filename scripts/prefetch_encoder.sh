#!/bin/bash
# Download a foundation-model encoder into the shared cache. Run on a LOGIN
# NODE, once, before submitting anything.
#
# Compute nodes on most clusters (Sherlock included) have no outbound internet,
# so the first from_pretrained() inside a job does not fail fast - it stalls
# until the connection times out, and a job array can burn every slot that way.
# Login nodes do have internet, and $SCRATCH is shared, so fetching here once
# leaves the weights readable from every compute node.
#
# Usage:
#     bash scripts/prefetch_encoder.sh              # phikon-v2, the default
#     bash scripts/prefetch_encoder.sh uni          # UNI (gated, needs login)
#     bash scripts/prefetch_encoder.sh owkin/phikon-v2

set -euo pipefail

MODEL=${1:-phikon}

# Same isolation the job scripts use: a stray package in ~/.local shadows the
# environment and has already broken transformers' imports once here.
export PYTHONNOUSERSITE=1
export HF_HOME=${HF_HOME:-${SCRATCH:-${HOME}}/huggingface}

mkdir -p "${HF_HOME}"

echo "cache   ${HF_HOME}"
echo "host    $(hostname)"

case "${MODEL}" in
    phikon) REPO="owkin/phikon-v2" ; LOADER="hf" ;;
    uni)    REPO="MahmoodLab/UNI"  ; LOADER="timm" ;;
    */*)    REPO="${MODEL}"        ; LOADER="hf" ;;
    *)      echo "unknown model: ${MODEL}" >&2 ; exit 1 ;;
esac

echo "model   ${REPO} (${LOADER})"

if [ "${LOADER}" = "timm" ]; then
    # UNI is gated. Without an accepted access request and a token this exits
    # with a 401 rather than anything more helpful, so say so up front.
    echo "note    gated model - run 'huggingface-cli login' first if this 401s"
    python -c "
import timm
timm.create_model('hf-hub:${REPO}', pretrained=True, num_classes=0)
print('cached')
"
else
    python -c "
from transformers import AutoModel
AutoModel.from_pretrained('${REPO}')
print('cached')
"
fi

echo
du -sh "${HF_HOME}"
echo
echo "Jobs can now run offline. In the job script, or already set by"
echo "scripts/train_chroma.sbatch:"
echo "    export HF_HOME=${HF_HOME}"
echo "    export HF_HUB_OFFLINE=1"
