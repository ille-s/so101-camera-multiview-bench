# Which camera is the policy actually using?

A multi-view policy sees several streams, and its success rate says nothing
about which of them it depends on. It may be leaning on one view and treating
the rest as decoration. This experiment finds out by degrading one camera at a
time, at inference only, and watching what the success rate does.

No retraining is involved. The same checkpoint is evaluated repeatedly, and
the only thing that changes is what one of its cameras shows it.

```{raw} html
<video autoplay loop muted playsinline width="100%" poster="../_static/experiments/exp_blackout.jpg">
  <source src="../_static/experiments/exp_blackout.mp4" type="video/mp4">
</video>
```

The wrist camera is zeroed here, and the policy keeps running on what is left.
Whether it still succeeds is the measurement.

## How it works

Degradation happens at the boundary where a rendered frame is handed to the
policy. The same transformed frame goes into the policy, the frame dumps and
the recorded videos, so the video is an honest record of what the policy saw.

Three flags control it:

| Flag | Meaning |
|---|---|
| `--augment_cameras` | Comma list of **policy** camera names to degrade, for example `wrist` or `front,top`. Nothing happens without it. |
| `--augment` | Which degradation. Default `blackout`. |
| `--augment_strength` | How strong, interpreted per mode (see below). Required for `gauss` and `blur`, which have no meaningful zero. |

Available modes, from `evaluation/augmentation.py`:

| Mode | What it does | Strength means |
|---|---|---|
| `blackout` | Zeroes the frame: total sensor outage | ignored |
| `gauss` | Adds Gaussian pixel noise | sigma on a 0 to 1 scale, so `0.05` is mild and `0.4` is severe |
| `blur` | Gaussian blur | odd kernel size, at least 3. `5` is mild, `31` is heavy |
| `invert` | Inverts colours | ignored |
| `contrast` | Stretches contrast around mid-grey | factor, above 1 is stronger |
| `canny` | Replaces the image with its edge map | upper Canny threshold |

The first three are the degradation family used for camera importance. The
others exist to ask a different question, which is what representation the
policy needs rather than how much signal.

What the three degradations do to a frame, at the strengths used above:

```{image} ../_static/experiments/augmentation_examples.jpg
:alt: The same wrist, front and top frames as original, blacked out, with Gaussian noise, and blurred
:width: 100%
```

Blackout removes the camera outright. Gaussian noise at sigma 0.15 leaves the
scene readable to a human but adds pixel-level noise the policy never saw in
training. Blur at kernel 15 keeps the layout and destroys the fine detail
around the gripper and the cube.

## Which combinations to run

Every run is one cell of a matrix, and each block of that matrix answers a
different question. For a policy with three cameras it comes to thirty cells:

| Block | Cells | Condition | The question it answers |
|---|---|---|---|
| Control | 1 | nothing degraded | What is this policy worth today, on this bin, with this seed? Everything else is measured against it. |
| Knockout | 3 | `blackout`, one camera at a time | Which camera is load-bearing? The bluntest cut there is. |
| Noise ladder | 9 | `gauss` at 0.05, 0.15, 0.40, one camera at a time | Does the policy need the camera, or does it merely prefer it? A camera that survives sigma 0.40 was barely being read. |
| Blur ladder | 9 | `blur` at kernel 5, 15, 31, one camera at a time | Is it the fine detail or the layout? Blur keeps where things are and removes what they look like. |
| Leave one out | 6 | `gauss` 0.40 and `blur` 31 on every camera *except* one | Can that one camera carry the task alone? |
| All at once | 2 | `gauss` 0.40 and `blur` 31 on all three | The floor. Whatever remains is not coming from the cameras. |

The count is 3 + 9n for n cameras, so two make twenty-one cells and four
make thirty-nine. The ladders are the
part worth keeping when the budget is tight, because a knockout on its own
cannot separate a camera the policy depends on from one it glances at.

Pick the bins with the same care. A single interior bin flatters the policy,
and camera importance measured only there does not have to hold in a corner
where the cube is hard to reach.

## Running it

Establish the baseline first, with no degradation at all, because the
comparison is against this number and not against an expectation:

```bash
# adjust: the policy, the bin it is evaluated on, and where results go
CKPT=outputs/3cam_wrist_front_top/checkpoints/065000/pretrained_model
BIN=datasets/01_raw_gamepad/bin_c2_r2
SCENE=src/so101_mvbench/tasks/scene_configs/lift_cube_6cam.json
OUT=outputs/3cam_wrist_front_top/camera_importance

python -m so101_mvbench.evaluation.async_eval \
    --mode async --num_envs 10 --num_episodes 27 \
    --policy_path "$CKPT" --scene_config "$SCENE" --dataset_root "$BIN" \
    --output_dir "$OUT/baseline" \
    --seed 42 --settle_steps 130 --full_report --headless
```

Then knock out one camera completely. Repeat once per camera the policy has:

```bash
# adjust: the policy, the bin it is evaluated on, and where results go
CKPT=outputs/3cam_wrist_front_top/checkpoints/065000/pretrained_model
BIN=datasets/01_raw_gamepad/bin_c2_r2
SCENE=src/so101_mvbench/tasks/scene_configs/lift_cube_6cam.json
OUT=outputs/3cam_wrist_front_top/camera_importance

python -m so101_mvbench.evaluation.async_eval \
    --mode async --num_envs 10 --num_episodes 27 \
    --policy_path "$CKPT" --scene_config "$SCENE" --dataset_root "$BIN" \
    --output_dir "$OUT/blackout_wrist" \
    --augment_cameras wrist --augment blackout \
    --seed 42 --settle_steps 130 --full_report --headless
```

A graded sweep says more than a knockout, because it separates a camera the
policy needs from one it merely prefers:

```bash
# adjust: the policy, the bin, where results go, and the camera under test
CKPT=outputs/3cam_wrist_front_top/checkpoints/065000/pretrained_model
BIN=datasets/01_raw_gamepad/bin_c2_r2
SCENE=src/so101_mvbench/tasks/scene_configs/lift_cube_6cam.json
OUT=outputs/3cam_wrist_front_top/camera_importance
CAM=wrist

for s in 0.05 0.15 0.40; do
    python -m so101_mvbench.evaluation.async_eval \
        --mode async --num_envs 10 --num_episodes 27 \
        --policy_path "$CKPT" --scene_config "$SCENE" --dataset_root "$BIN" \
        --output_dir "$OUT/gauss_${CAM}_${s}" \
        --augment_cameras "$CAM" --augment gauss --augment_strength "$s" \
        --seed 42 --settle_steps 130 --full_report --headless
done
```

Blur takes an odd kernel instead of a fraction:

```bash
# adjust: the policy, the bin, where results go, and the camera under test
CKPT=outputs/3cam_wrist_front_top/checkpoints/065000/pretrained_model
BIN=datasets/01_raw_gamepad/bin_c2_r2
SCENE=src/so101_mvbench/tasks/scene_configs/lift_cube_6cam.json
OUT=outputs/3cam_wrist_front_top/camera_importance
CAM=wrist

for k in 5 15 31; do
    python -m so101_mvbench.evaluation.async_eval \
        --mode async --num_envs 10 --num_episodes 27 \
        --policy_path "$CKPT" --scene_config "$SCENE" --dataset_root "$BIN" \
        --output_dir "$OUT/blur_${CAM}_${k}" \
        --augment_cameras "$CAM" --augment blur --augment_strength "$k" \
        --seed 42 --settle_steps 130 --full_report --headless
done
```

## Reading the result

Every run writes its own `eval_summary.json`, so the comparison is a table of
success rates against the baseline. Two cautions decide whether the table
means anything:

- **Repeat before concluding.** A single run's success rate carries the
  measurement's own noise. Run each cell the way the campaign does, several
  times with the same seed, and compare the difference against that spread.
  A drop smaller than the spread is not a finding.
- **Watch the videos of the degraded run.** The degradation reaches the
  recorded video, so you can see exactly what the policy was working with. A
  policy that fails under `blackout` on a camera it never used is a bug
  somewhere else.

The interesting outcome is asymmetry: if blacking out one camera costs most of
the success rate while the others cost little, that camera is carrying the
policy, and the others were along for the ride.

### What the matrix looked like when it was run

The numbers below come from the study this repository was cut from, so they
are one policy's answer and not a property of the method. The policy has the
three cameras `wrist`, `front` and `top`, each cell is 27 episodes at seed 42,
and each cell was run once.

Two bins were swept, because a single cube position cannot show whether camera
importance is a property of the policy or of where the cube happens to be: an
easy interior position, and the hardest corner.

The interior bin first:

```{image} ../_static/experiments/cam_importance_c2_r2.jpg
:alt: Success rate per degraded camera and the delta against the reference, interior bin
:width: 100%
```

One camera carries this policy. Blacking out the wrist takes it from a clean
100 percent to zero, and noise at sigma 0.40 on the wrist does the same
without removing a single pixel of the layout. The other two are close to
free *under noise*: front and top survive every strength the ladder offers,
and `top` under the heaviest blur still scores 100 percent. Blur is where the
wrist finally separates from the others, dropping to 14.8 percent at kernel 31
while front stays at 92.6.

Blacking a camera out is a different matter, and the heatmap is where to read
it: removing `front` outright still costs 39 points. So "the policy barely
reads front" holds for degraded pixels and not for no pixels at all, which is
the distinction the noise ladder exists to draw.

Two reference lines appear in the figure and they are not the same number. The
dashed one is the bin's success rate from the main evaluation campaign, ten
runs of the undegraded policy, and the heatmap deltas are measured against it.
The dotted one is the control cell of this sweep, a single run. Those two
differing by about six points is the measurement noise, drawn to scale.

The corner bin tells a different story:

```{image} ../_static/experiments/cam_importance_c4_r3.jpg
:alt: Success rate per degraded camera and the delta against the reference, corner bin
:width: 100%
```

Here every camera costs something. Blacking out `front` or `top` costs about
13 points each, where in the interior `top` cost 16 and `front` 39, and the
wrist no longer has a monopoly. What changed is not the policy but the
difficulty: at 70 percent to start with, there is less slack to absorb a
degraded view.

The leave-one-out block asks the sharper question, whether a camera can carry
the task by itself. Every camera except the named one is degraded:

| Only clean camera | interior, gauss 0.40 | interior, blur 31 | corner, gauss 0.40 | corner, blur 31 |
|---|---|---|---|---|
| `wrist` | 88.9 | 96.3 | 3.7 | 33.3 |
| `front` | 0.0 | 37.0 | 0.0 | 22.2 |
| `top` | 0.0 | 48.1 | 0.0 | 0.0 |
| none, all three degraded | 0.0 | 29.6 | 0.0 | 0.0 |
| *undegraded control* | *100.0* | *100.0* | *70.4* | *70.4* |

In the interior the wrist alone is worth almost the whole policy, 88.9 against
a control of 100. In the corner the same camera alone yields 3.7. So "the
wrist is the important camera" is true where the cube is easy to reach and
false where it is not, which is exactly the kind of claim a single bin would
have gotten wrong.

One honest caveat, the same one this page gives above: these cells were run
once each. A six-point difference between two of them is noise. A ninety-point
one is not, and every conclusion drawn here rests on the large drops.

## What this cannot tell you

It measures dependence of *this trained policy* on *this camera*, not the
value of the viewpoint in general. A policy trained without a camera might
have learned to use another one better. To ask that question, train separate
policies per camera subset and compare them ({doc}`../pipeline/03_training`).

## Where to go next

- {doc}`camera_drift` — the same idea for a camera that moved rather than degraded
- {doc}`../pipeline/04_evaluation` — the protocol these runs inherit
- {doc}`../pipeline/05_reporting` — turning the runs into numbers
