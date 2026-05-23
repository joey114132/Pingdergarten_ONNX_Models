#!/usr/bin/env bash
# Train all three wake words sequentially and collect the ONNX files
# into ./models/. Run from the repo root.
#
# Usage:
#   ./train_all.sh                # train all three
#   ./train_all.sh eduping        # train just one
#   SKIP_SETUP=1 ./train_all.sh   # skip background/RIR download if already done

set -euo pipefail

WAKEWORDS=("eduping" "nori_arm" "gogo_ping")
if [[ $# -gt 0 ]]; then
  WAKEWORDS=("$@")
fi

mkdir -p models

if [[ -z "${SKIP_SETUP:-}" ]]; then
  echo ">>> Downloading background noise + RIRs + VoxCPM2 weights (one-time setup)"
  # --config is required so setup pulls the VoxCPM2 snapshot. All three
  # configs share the same backend, so any one works here.
  livekit-wakeword setup --config "configs/${WAKEWORDS[0]}.yaml"
fi

for ww in "${WAKEWORDS[@]}"; do
  cfg="configs/${ww}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "!!! Missing config: $cfg" >&2
    exit 1
  fi

  echo
  echo "==================================================================="
  echo ">>> Training '$ww' from $cfg"
  echo "==================================================================="
  livekit-wakeword run "$cfg"

  # Trainer writes ONNX to <output_dir>/<model_name>/<model_name>.onnx, so the
  # nested model_name subdir is part of the actual path.
  src="output/${ww}/${ww}/${ww}.onnx"
  if [[ -f "$src" ]]; then
    cp "$src" "models/${ww}.onnx"
    echo ">>> Copied $src -> models/${ww}.onnx"
  else
    echo "!!! Expected ONNX not found at $src" >&2
    echo "    Check output/${ww}/ for the actual export path." >&2
    exit 1
  fi
done

echo
echo ">>> Done. ONNX models:"
ls -lh models/
