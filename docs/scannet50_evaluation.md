# ScanNet50 evaluation

`scripts/eval_scannet50.py` evaluates only the original DenseVGGT, FastVGGT,
and SelTR variants.  It uses the 50-scene list from FastVGGT's
`eval/scannet_50.yaml`.  The project's main protocol retains every third valid
RGB/pose/depth frame, then caps the selected sequence at `--num-frames`.
It rejects a scene with 300 or fewer valid source frames by default, preventing
the historical 300-frame cache from being used as if it were a full ScanNet
sequence.  Pass a complete processed ScanNet extraction via `--dataset-root`.

## FastVGGT fairness protocol

For a like-for-like FastVGGT comparison, use the separate fairness runner:

```bash
bash scripts/run_scannet50_100_300_fairness.sh /path/to/vggt.pt \
  /path/to/scannet50_processed /path/to/scannet_meshes 0,1,2,3,4,5,6
```

It passes `--fairness-fastvggt-protocol` to every method.  That switch uses
the released FastVGGT frame selector exactly: retain the first valid RGB/pose
frame, integer-stride sample the remaining valid RGB/pose frames, then truncate
at the requested count.  It intentionally does not alter the main protocol.
In this mode the FastVGGT baseline also uses the released protected bipartite
merge/unmerge ordering and camera/depth heads only; the existing latency timing
boundary is unchanged.

The unified runner accepts `FRAME_COUNTS` so the formal 500/1000-frame run is:

```bash
FRAME_COUNTS="500 1000" bash scripts/run_scannet50_fairness.sh /path/to/vggt.pt \
  /path/to/scannet50_processed /path/to/scannet_meshes 0,1,2,3,4,5,6 \
  outputs/scannet50_fairness_v3
```

It renders a live terminal progress bar over all expected scene jobs
(`50 × methods × requested frame counts`), including elapsed time and ETA.
Set `PROGRESS_INTERVAL=10` to update it every 10 seconds.

### Optional automatic shutdown after a rented-server experiment

The runner itself never shuts down a machine by default.  When running on a
rented server as root, append `/usr/bin/shutdown` to the *complete* fairness
command according to the desired failure policy:

```bash
# Shut down only after the complete experiment and its validation succeed.
FRAME_COUNTS="500 1000" bash scripts/run_scannet50_fairness.sh /path/to/vggt.pt \
  /path/to/scannet50_processed /path/to/scannet_meshes 0,1,2,3,4,5,6 \
  outputs/scannet50_fairness_v3 && /usr/bin/shutdown

# Shut down regardless of whether the experiment succeeds or fails.
FRAME_COUNTS="500 1000" bash scripts/run_scannet50_fairness.sh /path/to/vggt.pt \
  /path/to/scannet50_processed /path/to/scannet_meshes 0,1,2,3,4,5,6 \
  outputs/scannet50_fairness_v3 ; /usr/bin/shutdown
```

`&&` is recommended for formal experiments because it preserves the server for
debugging if a scene fails or the final 50-scene validation rejects the run.
The semicolon form is useful only when automatic shutdown is required even
after an error.

Run the formal configurations with a VGGT checkpoint:

```bash
bash scripts/run_scannet50_500.sh /path/to/vggt.pt
bash scripts/run_scannet50_1000.sh /path/to/vggt.pt
```

The launchers use GPUs 4, 5, and 6 for DenseVGGT, FastVGGT, and SelTR by
default.  Override them with `DENSE_GPU`, `FAST_GPU`, and `SELFTR_GPU`.  They
also default to the FastVGGT Python environment; set `EVAL_PYTHON` to use a
different environment.

The historical cache at
`/data_SSD1/mmc_lyxiang/dataset/scannet50_data/extracted/scannet-frames`
has only 300 frames per scene and is intentionally rejected by the evaluator.
Use a full extraction for every ScanNet50 experiment, including smoke tests:

```bash
bash scripts/test_scannet50_100.sh /path/to/vggt.pt \
  /path/to/scannet50_processed /path/to/scannet_meshes 0,1,2,3,4,5,6
```

This fairness smoke test evaluates `scene0000_00` and `scene0013_02`, each at
100 frames for all three methods. Scene jobs are assigned round-robin over the
provided GPU list. Because this smoke test has only two scenes, at most two
GPUs can be busy; the formal 50-scene runners distribute all scenes over every
provided GPU. It also validates the five visualization artifacts for every
method-scene pair; a missing PLY or PNG causes the smoke test to fail. It writes
`smoke_comparison.md` and `smoke_comparison.json`. The report keeps project
and FastVGGT reconstruction metrics in separate rows: project metrics are
compared only with the local project-protocol history, and FastVGGT-path CD is
shown alongside the FastVGGT paper's published 50-scene CD/latency values. Those external
references use different protocol scopes and are deliberately not treated as
pass/fail thresholds.

Each method writes per-scene `metrics.json` files and one aggregate
`metrics.json`.  Pose output includes AUC@3/5/15/20/30, RRA/RTA@5/30, ATE,
ARE, RPE rotation/translation, and APE.  AUC@330 in the request is interpreted
as AUC@30 and `RPW-trans` is output as `rpe_trans_rmse_m`.

Geometry follows FastVGGT's ScanNet route: estimated depth is unprojected with
estimated cameras, mapped back using the first GT camera, bbox-scale aligned to
the ScanNet mesh, then sampled and voxelized at 0.05 m.  Two independently
computed CD outputs are always written: `reconstruction.cd_m` is the project's
deterministic reservoir-sampled reference metric, while
`fastvggt_reconstruction.cd_m` exactly follows FastVGGT's full-cloud bbox
alignment followed by its concatenation-order 100k random sample.  Both are
clipped bidirectional sums `Acc + Comp`;
`overall_m` is `(Acc + Comp) / 2` for the reference metric.  The same output
includes the requested medians, normal consistency, precision/recall/F1 at
0.05 m, depth metrics, latency, FPS, peak allocated/reserved VRAM, and token
retention.  FastVGGT/DenseVGGT retain one fixed-policy statistic; SelTR records
all three grouping refresh stages.

With `--save-visualizations` (enabled by all three launcher scripts), each
scene also receives `visualization/reconstruction_overlay.ply`, separate
predicted/GT PLYs, `reconstruction.png`, and `trajectory_xz.png`. The
trajectory image uses FastVGGT's original `eval_trajectory(..., align=True)`
route: the first finite pose in the complete pose sequence as origin,
world-to-camera poses, the same Sim(3) alignment, EVO XZ plotting, GT dashed
trace, aligned APE colour map, and the same matplotlib-buffer/PIL PNG save
chain.
