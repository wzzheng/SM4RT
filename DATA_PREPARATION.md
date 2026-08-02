# Data Preparation

This document provides instructions for downloading and preparing the datasets used in this project.

## Download Instructions

### 1. Kubric

> **Note:** Kubric is a synthetic dataset generated using Blender. You can generate custom data via their public codebase, or download pre-generated splits.

**Source:** [Hugging Face Dataset](https://huggingface.co/datasets/your-org/kubric) *(replace with actual URL)*

```bash
# Install dependencies
pip install gdown huggingface_hub

# Download from Hugging Face
huggingface-cli download your-org/kubric \
  --local-dir ./data/kubric \
  --include "*.tfrecord" \
  --exclude "*.tmp"
```

**Expected structure:**
```
data/kubric/
├── train/
│   ├── scene_0001.tfrecord
│   ├── scene_0002.tfrecord
│   └── ...
├── val/
│   ├── scene_1001.tfrecord
│   └── ...
└── metadata/
    ├── object_categories.json
    └── split_info.json
```

---

### 2. Point Odyssey

**Source:** [Hugging Face Dataset](https://huggingface.co/datasets/your-org/point-odyssey) *(replace with actual URL)*

```bash
# Download via Hugging Face
huggingface-cli download your-org/point-odyssey \
  --local-dir ./data/point_odyssey \
  --include "*.hdf5" "*.mp4"
```

**Expected structure:**
```
data/point_odyssey/
├── train/
│   ├── sequence_001/
│   │   ├── frames/
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   ├── tracks.hdf5
│   │   └── metadata.json
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

---

### 3. Dynamic Replica

**Source:** [Hugging Face Dataset](https://huggingface.co/datasets/your-org/dynamic-replica) *(replace with actual URL)*

```bash
huggingface-cli download your-org/dynamic-replica \
  --local-dir ./data/dynamic_replica \
  --include "*.npz" "*.png"
```

**Expected structure:**
```
data/dynamic_replica/
├── scene_001/
│   ├── rgb/
│   │   ├── frame_0000.png
│   │   └── ...
│   ├── depth/
│   │   ├── frame_0000.png
│   │   └── ...
│   ├── flow/
│   │   ├── frame_0000.npz
│   │   └── ...
│   └── camera/
│       └── poses.npy
├── scene_002/
│   └── ...
└── splits/
    └── default.json
```

---

### 4. Stereo4D

**Source:** [Hugging Face Dataset](https://huggingface.co/datasets/your-org/stereo4d) *(replace with actual URL)*

```bash
huggingface-cli download your-org/stereo4d \
  --local-dir ./data/stereo4d \
  --include "*.png" "*.npy"
```

**Expected structure:**
```
data/stereo4d/
├── train/
│   ├── scene_000/
│   │   ├── left/
│   │   │   ├── 000000.png
│   │   │   └── ...
│   │   ├── right/
│   │   │   ├── 000000.png
│   │   │   └── ...
│   │   ├── disparity/
│   │   │   ├── 000000.npy
│   │   │   └── ...
│   │   └── calibration.json
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

---

## Data Preprocessing

After downloading all datasets, run the preprocessing script to unify formats:

```bash
python scripts/preprocess_data.py \
  --kubric_path ./data/kubric \
  --point_odyssey_path ./data/point_odyssey \
  --dynamic_replica_path ./data/dynamic_replica \
  --stereo4d_path ./data/stereo4d \
  --output_path ./data/processed
```

---

## Verification

To verify all datasets are correctly downloaded:

```bash
python scripts/verify_datasets.py \
  --datasets kubric,point_odyssey,dynamic_replica,stereo4d \
  --data_root ./data
```

Expected output:
```
✓ Kubric: 12,345 scenes found
✓ Point Odyssey: 8,760 sequences found
✓ Dynamic Replica: 42 scenes found
✓ Stereo4D: 15,000 stereo pairs found
All datasets verified successfully!
```

---

## Notes

- Total storage requirement: **~1.2 TB** (consider using symbolic links to external drives)
- All datasets are provided under their respective licenses — please check before use.
- For faster access, consider converting `.tfrecord` files to `.npy` or `.hdf5` formats.
- If you encounter download issues, try using `--resume-download` flag with `huggingface-cli`.

---

## References

- Kubric: [https://github.com/google-research/kubric](https://github.com/google-research/kubric)
- Point Odyssey: [https://pointodyssey.com/](https://pointodyssey.com/)
- Dynamic Replica: [https://dynamic-replica.github.io/](https://dynamic-replica.github.io/)
- Stereo4D: [https://stereo4d.github.io/](https://stereo4d.github.io/)