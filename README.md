# EigDiff for Preference-Based Diffusion Fine-Tuning

PCA-based data selection for fine-tuning text-to-image diffusion models
(Stable Diffusion 1.5 and PixArt-alpha) under Direct Preference Optimization
(DPO) and Supervised Fine-Tuning (SFT).

## Layout
```
SD/       Stable Diffusion 1.5 pipeline
PixArt/   PixArt-alpha pipeline (train_scripts/ holds the trainer + selection)
dataset_creation/  All scripts to get and adapt the data sets to the Diffusion-DPO format with the embedding script, used in this paper. 
```

## Environment
```
  We used Python 3.12.3
  python -m venv eigen-diffusion
  source eigen-diffusion/bin/activate
  pip install -r requirements.txt
```

## Dataset Names and How to Get Them
- **pick** — Pick-a-Pic v1 (pickapic-anonymous/pickapic_v1). The trainers take `--dataset_name pickapic-anonymous/pickapic_v1`, the other scripts `pick`.
- **hpdv2** — Run `unpack_hpdv2.py` to unzip tars (https://huggingface.co/datasets/ymhao/HPDv2/tree/main) + jsons from the link. `create_hpdv2.py` to get the set at `hpdv2_pickapic_format/`. Put `train.tar.gz`, `test.tar.gz` and `test.json` in `data/`, and `train.json` in `data/train/`.
- **rapidata** — `split_rapidata.py` (writes `train_rapidata.txt` /
  `test_rapidata.txt`) then `create_dataset_rapidata.py` to get the set at
  `processed_rapidata_dataset_bytes/`.
- **roco** — `create_roco.py` to create `roco_dataset/` (single-image,
  **SFT only**), `sample_roco.py` writes `roco_validation.txt` for eval.


## End-to-end Pipeline

1. **Create datasets** - previous step
2. **Build embeddings** (from the repo root; the same file serves both backbones):
   ```
   python dataset_creation/get_embeddings.py --dataset_name <dataset_name>  --seed 42 \
       --data_dir . --out_dir all_embeddings
   ```
   Pick-a-pic can be used with `cache_dir` instead.
   Pick, HPDv2, Rapidata Save `X` (chosen) and `Y` (rejected). ROCO saves `X` only.

   Coreset baselines are selected beforehand, once per dataset and seed, with the same `--first_eigen` and `--ratio` as training:
   ```
   python dataset_creation/get_baseline_features.py --dataset_name <dataset_name> --seed 42 --features pixels   # clip for semdedup
   python dataset_creation/select_baselines.py --dataset_name <dataset_name> --seed 42 --strategy kcenter
   ```
3. **Train** — edit and run the batch script, or call the trainer directly:
   ```
   ln -s ../all_embeddings ../baseline_indices .     # From SD path and from PixArt path
   chmod +x train_command.sh / chmod +x train_SD_batch.sh
   ./train_SD_batch.sh                 # From SD path
   ./train_command.sh              # From PixArt path
   ```
   For the local datasets built at the repo root, set `TRAIN_DATA_DIR=".."` in the batch script.
   Selection hyperparameters: `--first_eigen` (Stage-1 budget), `--ratio` (Stage-2 = ratio×n1),
   `--bin_count`, `--threshold` (τ1), `--thresh_stage2` (τ2, must be > τ1).
   Regime/ablation flags: `--sft` (SFT instead of DPO), `--random`
   (random-selection baseline, count-matched to the eigen subset), `--full`
   (full-scale baseline, no selection). **ROCO requires `--sft`.**
   Coreset baselines: `--strategy` (`eigen` by default, or `kcenter`, `facility_location`, `kmeans++`, `semdedup`, with the same budget).

   > The selection indices are into the tie-filtered, seed-shuffled (and, for
   > pick/hpdv2, 100k-capped) working set. The embeddings in step 2 must be built
   > with the **same `--seed`** as training, or the indices will not align (the
   > trainer raises an error if they don't).
4. **Inference** over a directory of trained runs
   (`<exp_dir>/<run>/checkpoint-N/lora/pytorch_lora_weights.safetensors`) For SD1.5
   (the PEFT adapter in `<exp_dir>/<run>/checkpoint-N/`) For PixArt
   ```
   python SD/inference.py       --dataset <dataset_name> --exp_dir RUNS [--checkpoint N]
   python PixArt/inference_pixart.py --dataset <dataset_name> --exp_dir RUNS [--checkpoint N]
   ```
   Writes one image per validation prompt + a `metadata.csv` per run, into `inference_<exp_dir>/` (SD1.5) or `generated_<exp_dir>/` (PixArt).
5. **Score** the generated images, from the folder step 4 ran in:
   ```
   python SD/pickscore.py   --paths <inference_dir>      # natural image datasets
   python SD/biomedclip.py  --paths <inference_dir>      # ROCO (medical)
   python SD/scores_unique.py --paths <inference_dir> --metric pickscore     # or biomedclip for ROCO
   ```
   `scores_unique.py` prints the scores for each run.
