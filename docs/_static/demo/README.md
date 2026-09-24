# Demo clips

Short looping clips embedded in the documentation. Each `.mp4` has a same-named
`.jpg` next to it, which is the `poster` frame the `<video>` tag shows before the
clip starts. Only `six_viewpoints` additionally has a `.gif`, because GitHub does
not render `<video>` in the repository README.

They were rendered outside this repository and are checked in as finished
artifacts. There is no generator script in this tree, so a clip cannot be
regenerated here — it can only be replaced.

| Clip | Size | What it shows | Embedded in |
|---|---|---|---|
| `six_viewpoints` | 960×526, gif 560×307 | One demonstration seen from six synchronized cameras: wrist, front, left, right, top, back | `docs/index.md`, repository `README.md` (gif) |
| `stage1_recording` | 960×766 | Four teleoperated episodes side by side, one per workspace-grid bin (`bin_c0_r0` and so on) | `docs/index.md`, `pipeline/01_recording.md` |
| `spawn_coverage` | 560×496 | The wrist camera across 540 demonstrations: the cube position and rotation of every episode | `docs/index.md`, `pipeline/01_recording.md` |
| `stage2_assembly` | 960×526 | The same trajectory re-rendered from six viewpoints | `docs/index.md`, `pipeline/02_assembly.md` |
| `stage3_training` | 960×526 | The same six-viewpoint grid, framed as the camera subset a policy is trained on | `docs/index.md`, `pipeline/03_training.md` |
| `stage4_evaluation` | 960×406 | A rollout of the trained policy, wrist next to external | `pipeline/04_evaluation.md` |
| `parallel_eval` | 1200×440 | Ten copies of the scene (`env0` … `env9`) stepping at once | `docs/index.md`, `pipeline/04_evaluation.md` |
| `eval_and_report` | 1100×452 | Left a failed rollout, right the task-progress histogram of the same bin | `docs/index.md`, `pipeline/05_reporting.md` |
| `stage5_reporting` | 920×506 | `stage_outcome.png`: episodes per task-progress stage | `pipeline/05_reporting.md` |

`six_viewpoints`, `stage2_assembly` and `stage3_training` are the same render —
identical 960×526, 277 frames, 9.23 s. They differ only in the title bar.

## The captions are burned in

The title bar at the top, and the footnote at the bottom where there is one
(`2x speed`, `sampled every 15th simulation step`, `wrist camera · 20 bins × 27
episodes`), are part of the pixels, not an overlay. Reusing a clip somewhere the
caption does not fit means cropping it.

`six_viewpoints_noheader.mp4` and `six_viewpoints_noheader.gif` are that crop,
kept alongside the originals. The docs embed the captioned versions; nothing
references the cropped ones.

The bar is 46 px on the 960-wide clips and 27 px on the scaled-down gif. Measure
before cropping a different clip: extract frame 0, and subtract the tile grid
height from the frame height.

```bash
# adjust: CLIP is the file, W x H is the tile grid below the bar, BAR its height
CLIP=six_viewpoints
W=960
H=480
BAR=46

ffmpeg -y -i "$CLIP.mp4" \
  -vf "crop=$W:$H:0:$BAR" \
  -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -movflags +faststart \
  "${CLIP}_noheader.mp4"
```

A gif needs its own palette pass, otherwise the grey gradient of the room bands:

```bash
# adjust: same three numbers as above, at the gif's own scale
CLIP=six_viewpoints
W=560
H=280
BAR=27

ffmpeg -y -i "$CLIP.gif" \
  -filter_complex "crop=$W:$H:0:$BAR,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3" \
  "${CLIP}_noheader.gif"
```
