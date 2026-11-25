# Sample Outputs

Three live showcase clips generated with **Edge0/Audio8-TTS-Realistic-Air (0.6B)** running on an NVIDIA GeForce RTX 3060 12GB (`bfloat16`, 44.1 kHz WAV):

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

## 3. Ye Grand ISP IVR — [`03-ye-grand-isp.wav`](./03-ye-grand-isp.wav)

- **Metadata**: [`03-ye-grand-isp.json`](./03-ye-grand-isp.json)
- **Audio Duration / Speed**: `50.93s` stitched (`49.88s` raw chunks + `350 ms` natural IVR pauses) generated in `39.64s` total model time / `39.96s` client wall time (`RTF 0.778`, `1074` frames, remote cold-start + tunnel startup `3.64s`)
- **Strategy**: The 676-character script exceeds upstream's recommended `<= 500` character per-utterance limit, so it was split into 4 natural IVR chunks. Chunk 1 establishes the speaker voice zero-shot, and Chunks 2–4 voice-clone Chunk 1 (`--reference-audio` + `--reference-text`) for seamless voice consistency.
- **Parameters**: `temperature=0.7`, `top_p=0.9`, `top_k=50`, `max_new_tokens=1024`, `bfloat16` on NVIDIA GeForce RTX 3060 12GB

| Chunk | Mode | Chars | Audio Duration | Gen Time | RTF | Frames | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Zero-shot | 134 | `9.71s` | `7.79s` | `0.803` | `209` | `1898844075` |
| 2 | Voice-clone (Chunk 1) | 196 | `15.65s` | `12.45s` | `0.795` | `337` | `1906309985` |
| 3 | Voice-clone (Chunk 1) | 194 | `15.00s` | `11.68s` | `0.779` | `323` | `1175645196` |
| 4 | Voice-clone (Chunk 1) | 118 | `9.52s` | `7.72s` | `0.811` | `205` | `1610243565` |

- **Prompt**:
> Ah, ladies and gentlemen, a most warm welcome to Ye Grand Internet Service Company—where modern marvels meet the finest customer care! If you'll kindly lend me your ear: Should your internet be on the fritz, press 1, and we shall mend it with haste! If your bill requires settling, press 2, for we do so appreciate a prompt patron. Craving unparalleled speed? Press 3, and we shall equip you with the swiftest connection of the age! Wishing to speak with a live representative? Press 0, and we shall connect you without delay! We thank you for your time and patronage—may your browsing be ever smooth and your signals ever strong! Ta-ta for now!
