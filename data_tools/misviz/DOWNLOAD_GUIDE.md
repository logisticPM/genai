# Misviz Data Download Guide

> Total download: ~5.5 GB
> Time estimate: 20-30 min (depending on internet speed)

---

## Step 1: Synth Data (images + data tables + axis metadata) — ~5 GB

This is the most important download — contains everything needed for the 2x2 experiment.

### 1a. Open TUdatalib

Open this URL in your browser:

```
https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/5003
```

### 1b. Download all 7z files

You will see 6 files:
- `data.7z.001` (1000 MB)
- `data.7z.002` (1000 MB)
- `data.7z.003` (1000 MB)
- `data.7z.004` (1000 MB)
- `data.7z.005` (1000 MB)
- `data.7z.006` (52 MB)

Click each file to download. Save them all to the same folder, e.g.:

```
C:\Users\chntw\Downloads\misviz_download\
```

### 1c. Extract

You need 7-Zip to extract (download from https://www.7-zip.org/ if needed).

Right-click `data.7z.001` -> 7-Zip -> Extract Here

It will automatically read all 6 parts and extract the full dataset.

### 1d. Move extracted files to project

After extraction you should see a folder structure like:

```
data/
├── png/                    # Chart images (81,814 files)
├── data_tables/            # CSV data tables
├── axis_data/              # JSON axis metadata
├── code/                   # Matplotlib code
└── ...
```

Copy these folders to:

```
C:\Users\chntw\Documents\7180\DD_v1\data\misviz_synth\
```

Final structure should be:

```
data\misviz_synth\
├── misviz_synth.json       # Already here (from git clone)
├── png\                    # From TUdatalib
├── data_tables\            # From TUdatalib
├── axis_data\              # From TUdatalib
└── code\                   # From TUdatalib
```

### 1e. Verify

Run:

```bash
cd C:\Users\chntw\Documents\7180\DD_v1
python -m data_tools.misviz.setup_data --check
```

All items should show `[OK]`.

---

## Step 2: Real-World Images — ~0.5 GB

These are the 2,604 real-world chart images (downloaded from original URLs).

### 2a. Run download script

```bash
cd C:\Users\chntw\Documents\7180\DD_v1\data\misviz
python download_images.py --use_wayback 0
```

This will download images from the original URLs. Some may fail (dead links).

### 2b. Retry with Wayback Machine

For any images that failed, retry with Internet Archive:

```bash
python download_images.py --use_wayback 1
```

### 2c. Check result

Downloaded images should be in:

```
data\misviz\img\
```

---

## Step 3: Verify Everything

```bash
cd C:\Users\chntw\Documents\7180\DD_v1
python -m data_tools.misviz.setup_data --check
python -m data_tools.misviz.setup_data --stats
```

Expected output:

```
[OK] Misviz JSON (real): ... (1454 KB)
[OK] Misviz-synth JSON: ... (38383 KB)
[OK] Misviz images dir: ... (2604 files)
[OK] Misviz-synth images dir: ... (57665+ files)
[OK] Misviz-synth data tables: ... (many files)
[OK] Misviz-synth axis metadata: ... (many files)
```

---

## Quick Test: Load One Instance

After download, verify data loading works:

```python
from data_tools.misviz.loader import MisvizLoader
from data_tools.misviz.text_context import TextContextBuilder

loader = MisvizLoader()
instance = loader.get_synth_instance(0)

print(f"Misleader: {instance.misleader}")
print(f"Chart type: {instance.chart_type}")
print(f"Has data table: {instance.data_table is not None}")
print(f"Has axis metadata: {instance.axis_metadata is not None}")

# Build text context for vision+text condition
builder = TextContextBuilder()
context = builder.build_context_from_instance(instance)
print(f"\nText context:\n{context}")
```

---

## What's Used Where

| Data | Used For | Required? |
|------|---------|-----------|
| `misviz_synth.json` | Instance metadata | Yes (already downloaded) |
| `png/` (synth images) | 2x2 experiment input | Yes |
| `data_tables/` | Vision+text condition | Yes (this is the text context) |
| `axis_data/` | OCR accuracy ground truth | Yes (for tool-use ablation) |
| `code/` | Debug / understanding | Optional |
| `misviz.json` | Real-world test metadata | Yes (already downloaded) |
| `img/` (real images) | Generalization test | Yes |
