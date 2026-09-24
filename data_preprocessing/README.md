# Dataset Format & Preprocessing (data_preprocessing/)

This is the first stage of the HOST pipeline (**dataset → alignment → coupling → policy_training**):
it defines the on-disk data convention that `alignment/` and `policy_training/` both consume, and
provides the tooling that constructs part of it (`task_paths.json`) from raw episode directories.
(`coupling/` is the second data-processing stage — it runs *after* `alignment/` training/eval and
produces `info_dtw.json`; see [`coupling/README.md`](../coupling/README.md).)

## 1. Task grouping

`build_task_dictionary.py` scans raw episode directories, groups them by task instruction, and
`write_task_paths.py` writes the resulting `task_paths.json` (`{"same": [...]}`) into each episode
directory. This is what lets `alignment/` sample a "Reference" episode of the same task for its
Main episode during TCC/DTW training, and what lets `policy_training/` sample a reference task
video for conditioning.

```bash
cd data_preprocessing
bash run_task_grouping.sh   # edit --input_dir / --output_path / --dataset_name inside first
```

`run_task_grouping.sh` composes two steps:

```bash
python build_task_dictionary.py \
  --input_dir /path/to/your/datasets \
  --output_path ./output/dataset_name.hdf5 \
  --dataset_name dataset_name \
  --clear

python write_task_paths.py \
  --hdf5_path ./output/dataset_name.hdf5
```

See `--help` on either script for the full argument list (`--skip_classes`, `--overwrite`,
`--dry_run`, `--max_paths`, ...). Raw data is expected in the layout described below, with an
additional session-level `instruction.json` mapping episode IDs to task instructions (used only by
`build_task_dictionary.py` to derive task groupings — not read by `policy_training/`/`alignment/`
directly).

Verification: this is a one-off, offline data-preprocessing script, not a training loop — verified
via static checks (`ast.parse`, CLI argument consistency) rather than the actual-run training
protocol used for `policy_training/`/`alignment/`.

## 2. Dataset format

`policy_training/` (self-grounded prediction) and `alignment/` (target coupling) consume the
**same on-disk data convention**: a JSON list of episode directory paths, each episode directory
following a shared layout, plus two shared sidecar mapping files (camera views, joint/action
normalization). This is the single source of truth for that format. Pass the episode-list JSON to
`alignment/train_scripts/run_ds.sh` through `VIDEO_PATHS`, and set the corresponding fields in
`policy_training/configs/data/custom*.yaml`.

Per-module differences are called out inline and summarized in [§2.6](#26-per-module-differences).

### 2.1. Video paths list — `{dataset_id}_video_paths.json`

A JSON array of episode directory paths. The dataset id is taken from the filename (e.g.
`10042_video_paths.json` → dataset id `"10042"`) and used as the lookup key into the camera
mapping, joint/action mapping, and any per-dataset config dicts (`dataset_fps`,
`dataset_image_size`, ...).

```json
["/data/episodes/task_a/episode_001", "/data/episodes/task_a/episode_002"]
```

- `policy_training/`: passed via `data.train.data_path` (comma-separated if multiple files); an
  `exclude_episode_json` field of the same format excludes listed episodes.
- `alignment/`: passed via `--video_paths` (comma-separated). Entries may also address a segment
  of a longer recording as `path:segment_id:start-end` (inclusive frame range), parsed by
  `AlignmentDataset._parse_video_path` in `alignment/datasets.py`.

### 2.2. Episode directory contents

```text
episode_001/
├── episode_001.json     # action/joint trajectory — required by policy_training/; optional for alignment/
├── instruction.txt      # plain-text task instruction — required by policy_training/; optional for alignment/
├── instruction.pt       # precomputed T5 text embedding {"context": [L,4096], "mask": [L]} — policy_training/ only
├── info_dtw.json        # {"aligned_progress": {"<frame_idx>": <0..1>, ...}} — policy_training/ only, optional
├── task_paths.json      # {"same": ["/data/episodes/task_a/episode_002", ...]} — peer episodes of the same task
├── images/              # frame sequence for one camera view: 0.jpg, 1.jpg, ...
└── gripper_images/      # frame sequence for a second camera view (if 2-view)
```

Video files (`{view}.mp4`) are also supported instead of an image-sequence subdirectory. Frame
count must match the number of entries in `episode_001.json`'s `"data"` array where that file is
used — no silent truncation.

`task_paths.json` and `info_dtw.json` aren't hand-authored — `task_paths.json` is produced by
[§1 above](#1-task-grouping), and `info_dtw.json` is produced by
[`coupling/`](../coupling/README.md) from `alignment/`'s own DTW training/eval output.

`task_paths.json` lists peer episodes of the *same task*. `policy_training/` uses it to sample a
reference task video shown alongside the primary observation; `alignment/` uses it to pick the
"Reference" video that the "Main" video's chunks are aligned against (the `"same"` pool is
preferred, a lower-quality `"100-95"` pool is a fallback if present). Without it, `alignment/`
falls back to self-alignment (Main and Reference are the same episode).

`episode_001.json` (action/joint trajectory):

```json
{
  "data": [
    {"follow_left_position": [x, y, z], "follow_left_rotation": [r, p, y], "follow_left_gripper": g,
     "follow_right_position": [x, y, z], "follow_right_rotation": [r, p, y], "follow_right_gripper": g},
    "... one entry per frame ..."
  ]
}
```

Rotations are `[roll, pitch, yaw]` Euler triples; `policy_training/` converts these to a 6D
rotation-matrix representation during loading when `use_6d_rotation: true`.

### 2.3. Camera mapping — `{cam_mapping_dir}/{dataset_id}_cam_mapping.json`

Maps each task directory (one level above the episode directory) to its ordered list of camera/
view names on disk:

```json
{"/data/episodes/task_a": ["images", "gripper_images"]}
```

- `policy_training/`: `num_view_probs` (e.g. `'{"2": 1.0}'`) selects how many of the listed views
  to sample per batch.
- `alignment/`: only needed for multi-view episodes (`CONFIG.DATA.MULTI_VIEW_DATASETS` /
  `CONFIG.DATA.NUM_VIEWS`); with a single view this file can be omitted (the first available
  camera / `images/` subdir is used).

### 2.4. Joint/action normalization — `{joint_action_mapping_dir}/{dataset_id}_joint_action_mapping.json`

Declares the ordered input-state fields (`joint_keys`), output-action fields (`action_keys`),
and per-field `min` / `delta` parameters. The file has an outer dataset-path mapping:

```text
{
  "<dataset path prefix>": {
    "action_keys": [...],
    "joint_keys": [...],
    "norm_min_delta": {"<field>": {"min": [...], "delta": [...]}}
  }
}
```

See the complete [mapping example](../格式说明/10042_joint_action_mapping.json) and
[action-format guide (中文)](../格式说明/README.md) for the dual-arm field order, 6D rotations,
relative actions, normalization formulas, and entry-selection behavior. Ranges can differ
between datasets; the example is not a universal checkpoint normalization configuration.
Use the mapping associated with training when running inference.

Both modules configure the directory containing these files through
`joint_action_mapping_dir` / `JOINT_ACTION_MAPPING_DIR`.

### 2.5. Remaining per-dataset config fields (`policy_training/` only)

In `policy_training/configs/data/custom.yaml`, also set per dataset id: `dataset_fps` (used for
minimum-length filtering and action-frame sampling) and `dataset_image_size` (`[width, height]`).
If your dataset has a different action dimensionality than the shipped example, update
`processor.action_output_dim` / `processor.proprio_output_dim` and the corresponding `action_dim`
in your model config to match.

### 2.6. Per-module differences

| | `policy_training/` | `alignment/` |
|---|---|---|
| Action/joint data (`episode_001.json`) | **Required** | Optional — only read if `CONFIG.JOINTS.USE_JOINTS: true` |
| `instruction.txt` / `instruction.pt` | Required | Optional |
| `info_dtw.json` (progress) | Used, optional | Not used |
| `task_paths.json` | Used for reference-task sampling | Used for Main/Reference pairing (falls back to self-alignment if absent) |
| Camera mapping | Used for multi-view sampling | Only needed for multi-view episodes |
| Config location | `policy_training/configs/data/custom*.yaml` | `alignment/config.py` (`CONFIG.DATA.*`, `CONFIG.JOINTS.*`) |
