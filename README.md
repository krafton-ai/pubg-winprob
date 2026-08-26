# PUBG Win Probability Prediction

Real-time win probability prediction for PUBG esports matches using Transformer-based survival analysis. Given in-game features (combat stats, positions, zone dynamics) at any point during a match, the model predicts each squad's probability of winning (WWCD - Winner Winner Chicken Dinner).

## Project Structure

```
pubg-winprob/
├── src/
│   ├── data/                  # Data loading & feature definitions
│   │   ├── continuous_features.py   # Continuous feature definitions
│   │   ├── dataset.py               # PyTorch Dataset
│   │   ├── dataset_v2.py            # Dataset with on-the-fly zone distance features
│   │   └── utils.py                 # Position/zone parsing utilities
│   ├── models/                # Neural network architecture
│   │   ├── backbone.py              # Transformer backbone (+ MLP encoder ablation option)
│   │   ├── modules.py               # Fourier encoding, zone/token embeddings
│   │   └── heads.py                 # Prediction heads (survival, hazard, classification)
│   ├── training/              # Training & evaluation pipeline
│   │   ├── trainer.py               # Main training entry point (DDP multi-GPU support)
│   │   ├── losses.py                # Loss functions (MSE, Cox, Ranking, CE, etc.)
│   │   ├── metrics.py               # Accuracy, C-index, IBS, ECE, log loss
│   │   ├── evaluation.py            # Phase-wise test evaluation
│   │   ├── inference.py             # Checkpoint loading & batch inference
│   │   └── calibration.py           # Temperature scaling for probability calibration
│   └── baselines.py           # Table 1 baseline implementations
├── scripts/
│   ├── run_experiments.sh           # Hyperparameter grid search (calls src.training.trainer)
│   ├── run_baselines.py             # Table 1 baseline runner (9 baselines)
│   └── run_inference.sh             # Inference entry point
├── data_generation/
│   ├── feature_engineering/         # Feature extraction from match logs
│   ├── generate_postmatch_dataset/  # Post-match JSON → CSV pipeline
│   └── generate_competitive_dataset/# Competitive match dataset generation
├── notebooks/
│   ├── run_files/                   # Training pipeline notebooks
│   ├── analysis/                    # Model evaluation & comparison
│   ├── calibration/                 # Temperature scaling notebooks
│   └── inference/                   # Inference demos
└── requirements.txt
```

## Model Architecture

- **Transformer backbone** with multi-head self-attention over squads (up to 16 per match)
- **Fourier positional encoding** for 3D player coordinates
- **Zone embedding** for bluezone/whitezone position and radius
- **Map embedding** for 4 PUBG maps (Erangel, Miramar, Taego, Rondo)
- **Multiple prediction heads**: survival time regression, Cox hazard, winner classification
- **Encoder ablation**: `--encoder mlp` uses per-squad MLPs with no cross-squad attention, under identical features, losses, and training settings

## Supported Loss Functions

| Loss Type | Description |
|---|---|
| `mse` | Survival time regression |
| `cox` | Cox partial likelihood (full retrospective) |
| `rank_cox` | Cox + ranking consistency penalty |
| `weighted_cox` | Elimination + survival likelihood with early-focus time weighting (paper loss) |
| `ce` | Cross-entropy classification |
| `concordance` | C-index with time-gap weighting |
| `survival_ce` | Survival score cross-entropy |

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch >= 2.0.0
- pandas, numpy, scipy, tqdm
- scikit-learn, lightgbm, scikit-survival, pycox, torchtuples (baselines)
- captum (for GradientSHAP analysis)
- matplotlib, seaborn (visualization)

## Usage

### Reproducing the paper

The paper model is trained with the following configuration:

```bash
python -m src.training.trainer \
    --folder_path /path/to/data --split_csv_path /path/to/data/split_files.csv \
    --embed_dim 64 --num_heads 8 --num_layers 8 --dropout 0.1 \
    --batch_size 512 --lr 1e-4 --weight_decay 1e-4 \
    --num_epochs 10 --patience 4 \
    --loss_type weighted_cox --seed 42
```

Optimizer is AdamW with early stopping on validation loss (patience 4). The
feed-forward hidden dimension is `embed_dim x 4 = 256` and the Fourier
positional encoding uses 8 frequency bands. Use `--seed` to run multi-seed
replicates and `--encoder mlp` for the encoder ablation.

After training, calibrate the winner probabilities with temperature scaling
(grid search over tau in [0.1, 2.0] with step 0.01 followed by local
refinement, minimizing validation ECE with B=15 bins):

```bash
python -m src.training.calibration \
    --checkpoint_path checkpoints/<exp_name>/best.pt \
    --folder_path /path/to/data --split_csv_path /path/to/data/split_files.csv
```

### Training

Main entry point is `src.training.trainer` (see above). Use `--toy_ratio 0.1`
for quick testing with a subset of data, and `--world_size N` for multi-GPU
DDP training.

### Hyperparameter Search

```bash
bash scripts/run_experiments.sh
```

Runs a grid search over embedding dimensions (64/128/256), attention heads
(4/8), layers (4/6/8), dropout (0.1/0.2), and learning rates (1e-3/1e-4);
the grid includes the paper configuration.

### Baselines (Table 1)

All nine baselines share the Transformer's split and evaluation protocol
(grouped by `(match_id, time_point)`, scored with the same metric code):

```bash
python scripts/run_baselines.py \
    --folder_path /path/to/data \
    --split_csv_path /path/to/data/split_files.csv \
    --models RandomGuess,RuleBased,LogReg,LogHaz,RFClf,LGBMReg,LGBMClf,CoxPH,DeepSurv \
    --seed 42
```

Add `--calibrate` for per-model temperature scaling on the validation set
(same tau grid and ECE bins as the main model).

Each baseline's key hyperparameters are exposed as CLI flags; the search
grids used for the paper are:

| Baseline | Hyperparameter grid |
|---|---|
| LogReg | `C` ∈ {0.01, 0.1, 1, 10} |
| RFClf | `n_estimators` ∈ {100, 300}, `max_depth` ∈ {10, 20, None} |
| LGBMReg / LGBMClf | `num_leaves` ∈ {31, 63, 127}, `learning_rate` ∈ {0.01, 0.05, 0.1}, `n_estimators` ∈ {100, 500, 1000} |
| CoxPH | `alpha` (L2) ∈ {0.1, 1, 10} |
| DeepSurv / LogHaz | `lr` ∈ {1e-3, 1e-4}, epochs ≤ 20 with early stopping |
| LogHaz | `num_durations` ∈ {25, 50, 100} |

### Inference

```bash
# Single CSV
bash scripts/run_inference.sh \
    --checkpoint_dir ./checkpoints/exp_001 \
    --csv_path ./test_data.csv

# Batch inference on folder
bash scripts/run_inference.sh \
    --checkpoint_dir ./checkpoints/exp_001 \
    --folder_path ./test_folder \
    --file_list "match1.csv match2.csv"
```

## Data Format

Each match is represented as a CSV with **192 phase-sampled time points**:
178 time points drawn per phase with a non-uniform density that grows toward
the late game (~60s intervals in phase 1 down to ~5s in phases 7-10), plus 14
fixed phase-boundary anchor points shared across matches. Phase membership of
a time point is determined by its time value (`src/data/utils.py`).

Features include:

- **Action features**: kills, damage dealt (by weapon type), healing, pickups
- **Spatial features**: player positions, distance to bluezone/whitezone
- **Squad status**: alive count, total health, armor/backpack levels
- **Zone dynamics**: bluezone/whitezone center and radius

Squads are padded to a maximum of 16 per match.

## Evaluation Metrics

Primary metrics reported in the paper:

- **Winner Accuracy**: whether the predicted winner matches the actual winner
- **C-index**: concordance between predicted and actual survival-time ranking
- **IBS (Integrated Brier Score)**: squared error of winner probabilities summed over the risk set and averaged across time

Calibration diagnostics:

- **ECE (Expected Calibration Error)**: reliability of probability estimates (B=15 confidence bins)
- **Log Loss**: cross-entropy of predicted win probabilities

All metrics are averaged over the phase-sampled time points; phase-wise
breakdowns over the 10 match phases are also reported.

## License

Apache License 2.0
