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
bash scripts/test_scannet50_100.sh /path/to/vggt.pt
```

Point `SCANNET_DATA_ROOT` to a full-frame ScanNet extraction before a run; the
source-pool guard and `--require-exact-frames` prevent an accidental partial
sequence result.

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
predicted/GT PLYs, `reconstruction.png`, and `trajectory_xz.png`.  The
trajectory uses FastVGGT's XZ projection and colours the aligned estimated
trajectory by absolute position error.
