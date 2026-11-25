# Sample Outputs

Two live showcase clips generated with **Edge0/Audio8-TTS-Realistic-Air (0.6B)** running on an NVIDIA GeForce RTX 3060 12GB (`bfloat16`, 44.1 kHz WAV):

## 1. Zero-Shot Default Voice — [`01-zero-shot.wav`](./01-zero-shot.wav)

- **Metadata:** [`01-zero-shot.json`](./01-zero-shot.json)
- **Duration / Speed:** `15.19s` audio generated in `11.77s` (`RTF 0.775`, `327` frames, seed `559131312`)
- **Parameters:** `temperature=0.75`, `top_p=0.9`, `top_k=50`, `max_new_tokens=1024`
- **Prompt:**
  > Wait-seriously? You unlocked an RTX 3060 on NixOS, wired up a fresh neural speech engine from scratch, and now it's talking straight through your headset? Honestly, that's pretty awesome.

Reproduce locally:

```bash
python infer.py \
  --text "Wait-seriously? You unlocked an RTX 3060 on NixOS, wired up a fresh neural speech engine from scratch, and now it's talking straight through your headset? Honestly, that's pretty awesome." \
  --temperature 0.75 --top-p 0.9 --top-k 50 --seed 559131312 \
  --output out-01.wav
```

## 2. Zero-Shot Voice Clone — [`02-voice-clone.wav`](./02-voice-clone.wav)

- **Metadata:** [`02-voice-clone.json`](./02-voice-clone.json)
- **Reference Clip:** [`reference-chicanery.wav`](./reference-chicanery.wav)
- **Reference Transcript:**
  > And I saved him. And I shouldn't have. I took him into my own firm.
- **Duration / Speed:** `14.91s` audio generated in `12.17s` (`RTF 0.817`, `321` frames, seed `1165230192`)
- **Parameters:** `temperature=0.7`, `top_p=0.9`, `top_k=50`, `max_new_tokens=1024`
- **Prompt:**
  > You think this is something? You think this is bad? This-this chicanery! He unlocked the GPU firmware, built a NixOS flake, and ran a zero-shot voice clone right over SSH! And it sounds completely real!

Reproduce locally:

```bash
python infer.py \
  --reference-audio samples/reference-chicanery.wav \
  --reference-text "And I saved him. And I shouldn't have. I took him into my own firm." \
  --text "You think this is something? You think this is bad? This-this chicanery! He unlocked the GPU firmware, built a NixOS flake, and ran a zero-shot voice clone right over SSH! And it sounds completely real!" \
  --temperature 0.7 --top-p 0.9 --top-k 50 --seed 1165230192 \
  --output out-02.wav
```
