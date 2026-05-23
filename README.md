# Robot Wake Word ONNX Extractor (Korean)

Trains three custom Korean wake word detectors using
[livekit/livekit-wakeword](https://github.com/livekit/livekit-wakeword) and
exports one ONNX file per phrase:

| Wake phrase | Hangul | Output |
| --- | --- | --- |
| eduping   | 에듀핑 | `models/eduping.onnx` |
| nori arm  | 노리암 | `models/nori_arm.onnx` |
| gogo ping | 고고핑 | `models/gogo_ping.onnx` |

Each model is a binary detector for its own phrase, so at runtime you load
all three and check which one fires.

## 1. System dependencies

```bash
sudo apt install espeak-ng libsndfile1 ffmpeg sox portaudio19-dev
```

## 2. Python install

Korean wake words use the **VoxCPM2** TTS backend (Piper is English-only).
Make sure the `voxcpm` extra is included:

```bash
python -m venv .venv && source .venv/bin/activate
pip install "livekit-wakeword[train,eval,export,voxcpm]"
```

GPU recommended. 50k steps on a small `conv_attention` model takes a few
hours on a single consumer GPU. VoxCPM also runs on GPU during the
data-generation stage and is slower than Piper because it synthesizes
sequentially.

## 3. (Optional) Sanity-check the synthetic Korean audio

VoxCPM2 doesn't have an `espeak-ng`-style preview, but the pipeline writes
generated clips into `output/<name>/generated/` after the `generate`
stage. To listen before committing to 50k training steps, you can run the
generate stage solo with a tiny `n_samples`:

```bash
# Edit a config so n_samples: 50 and n_samples_val: 10, then:
livekit-wakeword generate configs/eduping.yaml
# Play a few clips from output/eduping/generated/
```

If the pronunciation is wrong, adjust the Hangul spelling in
`target_phrases` (e.g. add a space: `"에듀 핑"`) and re-generate. Keep
`model_name` unchanged — it controls the ONNX filename.

## 4. Train

```bash
./train_all.sh                  # all three, sequential
./train_all.sh eduping          # one at a time
SKIP_SETUP=1 ./train_all.sh     # skip background/RIR/VoxCPM re-download
```

The script:
1. Runs `livekit-wakeword setup --config configs/eduping.yaml` once to
   fetch background noise, room impulse responses, **and** the VoxCPM2
   model snapshot from Hugging Face into `./data/`.
2. Runs `livekit-wakeword run configs/<name>.yaml` for each wake word.
   Each run writes to its own `./output/<name>/` directory so artifacts
   don't collide.
3. Copies the exported `<name>.onnx` into `./models/`.

## 5. Layout after training

```
configs/
  eduping.yaml
  nori_arm.yaml
  gogo_ping.yaml
data/
  backgrounds/        # MUSAN subset (shared)
  rirs/               # room impulse responses (shared)
  voxcpm/VoxCPM2/     # TTS weights (shared, ~GBs)
output/
  eduping/            # generated audio, features, checkpoints, eduping.onnx
  nori_arm/
  gogo_ping/
models/
  eduping.onnx        # the files you ship to the robot
  nori_arm.onnx
  gogo_ping.onnx
train_all.sh
```

## Tuning knobs (per-config)

- `n_samples`: bump to 30000–50000 if real-world false-positive rate is
  too high.
- `steps`: 50000 here; raise to 100000 for the prod-grade setup.
- `target_fp_per_hour`: lower = fewer false triggers, may hurt recall.
- `max_negative_weight`: raise to push the model harder against
  adversarial negatives.
- `custom_negative_phrases`: the auto-generated adversarial negatives in
  livekit-wakeword are English/CMUdict-biased, so for Korean the manual
  list carries the load. Add anything the model false-triggers on once
  you start field-testing.
- `model.model_size`: `small` is sized for embedded/robot use. Use `tiny`
  if you need to fit on a microcontroller, `medium` if accuracy matters
  more than latency.
- `voxcpm_tts.voice_design_prompts`: the configs use Korean-targeted
  persona prompts. Edit/extend these to broaden speaker variety.
