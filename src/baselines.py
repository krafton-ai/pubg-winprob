"""
Baseline models for PGC winner prediction (Table 1 of the paper).

Implemented baselines:
- RandomGuess: pick one random alive squad as winner
- RuleBased:   surviving members + total squad health, softmax-normalized
- LogReg:      logistic regression on binary winner label (scikit-learn)
- LogHaz:      LogisticHazard / Nnet-survival, discrete-time hazard (pycox)
- RFClf:       random forest classifier on binary winner label (scikit-learn)
- LGBMReg:     LightGBM regression on squad_death_time
- LGBMClf:     LightGBM binary winner classifier
- CoxPH:       Cox proportional hazards on remaining time (scikit-survival)
- DeepSurv:    Cox partial likelihood with an MLP encoder (pycox)

All models share the same evaluation protocol as the Transformer: predictions
are grouped by (match_id, time_point) and scored with
``src.training.metrics.compute_all_metrics``.

Survival-type baselines (CoxPH, DeepSurv, LogHaz) use conditional remaining
time prediction: each alive (match, time_point, squad) observation is a
sample with target ``remaining_time = squad_death_time - time_point`` and
event indicator 1 if the squad was eliminated (0 = winner, censored).

References:
- Cox (1972): Regression Models and Life-Tables
- Katzman et al. (2018): DeepSurv
- Gensheimer & Narasimhan (2019): Nnet-survival (LogisticHazard)
"""

import importlib.util
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler

# Optional dependencies — each baseline is skipped gracefully when its
# library is unavailable.
HAS_SKSURV = importlib.util.find_spec('sksurv') is not None
if HAS_SKSURV:
    from sksurv.linear_model import CoxPHSurvivalAnalysis

HAS_PYCOX = (
    importlib.util.find_spec('pycox') is not None
    and importlib.util.find_spec('torchtuples') is not None
)
if HAS_PYCOX:
    from pycox.models import CoxPH as PycoxCoxPH, LogisticHazard
    import torchtuples as tt

HAS_LGBM = importlib.util.find_spec('lightgbm') is not None
if HAS_LGBM:
    import lightgbm as lgb


def set_seed(seed: int):
    """Seed all random number generators for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Common preprocessing
# =============================================================================

def preprocess_features(
    X: np.ndarray,
    fit_scaler: bool = True,
    scaler: Optional[StandardScaler] = None,
    valid_features: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, StandardScaler, np.ndarray]:
    """
    Common feature preprocessing: standardization and zero-variance removal.

    Returns:
        X_clean: Preprocessed features (n_samples, n_valid_features)
        scaler: Fitted StandardScaler
        valid_features: Boolean mask for retained features
    """
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        feature_std = np.std(X_scaled, axis=0)
        valid_features = feature_std > 1e-8
        X_clean = X_scaled[:, valid_features]
    else:
        if scaler is None or valid_features is None:
            raise ValueError("scaler and valid_features are required when fit_scaler=False")
        X_clean = scaler.transform(X)[:, valid_features]

    return X_clean, scaler, valid_features


def prepare_remaining_time_data(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for conditional remaining-time survival models.

    For each (match, time_point, squad) observation where the squad is alive:
    - X: features at the current time point
    - remaining_time: squad_death_time - time_point
    - event: 1 if eliminated (not winner), 0 if winner (censored)
    """
    alive_mask = df['time_point'] < df['squad_death_time']
    alive_df = df[alive_mask]

    if len(alive_df) == 0:
        return np.array([]), np.array([]), np.array([])

    remaining_time = alive_df['squad_death_time'].values - alive_df['time_point'].values
    events = (alive_df['squad_win'] == 0).astype(float).values
    X = alive_df[feature_cols].values

    return X.astype(np.float32), remaining_time.astype(np.float32), events.astype(np.float32)


def _classifier_feature_cols(feature_cols: List[str], df: pd.DataFrame) -> List[str]:
    """Feature columns for the classifier/regressor family (adds time_point)."""
    cols = feature_cols.copy()
    if 'time_point' not in cols and 'time_point' in df.columns:
        cols.append('time_point')
    excluded = {'phase', 'match_id', 'squad_number', 'squad_win',
                'squad_death_time', 'squad_ranking'}
    return [f for f in cols if f not in excluded]


def _sort_deterministic(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows deterministically for reproducible training."""
    keys = [c for c in ('match_id', 'time_point', 'squad_number') if c in df.columns]
    if keys:
        return df.sort_values(keys).reset_index(drop=True)
    return df


# =============================================================================
# Survival-type baselines
# =============================================================================

def train_coxph_baseline(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    alpha: float = 10.0,
    seed: int = 42,
) -> Optional[Any]:
    """
    Cox Proportional Hazards on conditional remaining time (scikit-survival).

    Hyperparameter search grid used for the paper:
        alpha (L2 regularization): {0.1, 1.0, 10.0}
    """
    if not HAS_SKSURV:
        print("  [SKIP] scikit-survival not available")
        return None

    set_seed(seed)
    train_df = _sort_deterministic(train_df)

    X, remaining_time, events = prepare_remaining_time_data(train_df, feature_cols)
    if len(X) == 0:
        print("  [ERROR] No valid training data")
        return None

    valid_mask = remaining_time > 0
    X, remaining_time, events = X[valid_mask], remaining_time[valid_mask], events[valid_mask]

    X_clean, scaler, valid_features = preprocess_features(X, fit_scaler=True)
    if X_clean.shape[1] == 0:
        print("  [ERROR] No valid features after preprocessing")
        return None

    # Clip standardized features to keep exp(X @ beta) numerically stable.
    X_clean = np.clip(X_clean, -3.0, 3.0)

    y_struct = np.empty(len(events), dtype=[('event', bool), ('time', float)])
    y_struct['event'] = events.astype(bool)
    y_struct['time'] = remaining_time.astype(float)

    print(f"  Training CoxPH on {len(X_clean)} samples, {X_clean.shape[1]} features...")
    model = CoxPHSurvivalAnalysis(alpha=alpha)
    model.fit(X_clean, y_struct)

    model.feature_mask_ = valid_features
    model.scaler_ = scaler
    model.feature_cols_ = feature_cols
    return model


def get_deepsurv_net(in_features: int) -> torch.nn.Module:
    """DeepSurv network: 128 -> 64 MLP with BatchNorm and dropout."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, 128),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(128),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(64),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 1),
    )


def train_deepsurv_baseline(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    val_df: pd.DataFrame = None,
    num_epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> Optional[Any]:
    """
    DeepSurv (pycox CoxPH) on conditional remaining time.

    Hyperparameter search grid used for the paper:
        lr: {1e-3, 1e-4}, batch_size: {256, 512}, epochs up to 20
        (early stopping on validation loss, patience 5)
    """
    if not HAS_PYCOX:
        print("  [SKIP] pycox not available")
        return None

    set_seed(seed)
    train_df = _sort_deterministic(train_df)

    X_train, remaining_time, events = prepare_remaining_time_data(train_df, feature_cols)
    if len(X_train) == 0:
        print("  [ERROR] No valid training data")
        return None

    X_clean, scaler, valid_features = preprocess_features(X_train, fit_scaler=True)
    if X_clean.shape[1] == 0:
        print("  [ERROR] No valid features after preprocessing")
        return None

    in_features = X_clean.shape[1]
    print(f"  Training DeepSurv on {len(X_clean)} samples, {in_features} features (device: {device})...")

    net = get_deepsurv_net(in_features).to(device)
    model = PycoxCoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(lr)

    x_train = X_clean.astype('float32')
    y_train = (remaining_time.astype('float32'), events.astype('float32'))

    val_data = None
    if val_df is not None:
        val_df = _sort_deterministic(val_df)
        X_val, rt_val, ev_val = prepare_remaining_time_data(val_df, feature_cols)
        if len(X_val) > 0:
            X_val_clean, _, _ = preprocess_features(
                X_val, fit_scaler=False, scaler=scaler, valid_features=valid_features
            )
            val_data = (X_val_clean.astype('float32'),
                        (rt_val.astype('float32'), ev_val.astype('float32')))

    best_val_loss = float('inf')
    patience, patience_counter = 5, 0
    for epoch in range(num_epochs):
        log = model.fit(x_train, y_train, batch_size=batch_size, epochs=1,
                        val_data=val_data, verbose=False)
        if val_data is not None and hasattr(log, 'to_pandas'):
            log_df = log.to_pandas()
            if len(log_df) > 0 and 'val_loss' in log_df.columns:
                current = log_df['val_loss'].iloc[-1]
                if current < best_val_loss:
                    best_val_loss, patience_counter = current, 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break

    model.compute_baseline_hazards()
    model.feature_mask_ = valid_features
    model.scaler_ = scaler
    model.feature_cols_ = feature_cols
    return model


def train_loghaz_baseline(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    num_epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    num_durations: int = 50,
    seed: int = 42,
) -> Optional[Any]:
    """
    LogisticHazard (Nnet-survival) discrete-time hazard model (pycox).

    Hyperparameter search grid used for the paper:
        lr: {1e-3, 1e-4}, num_durations: {25, 50, 100}
    """
    if not HAS_PYCOX:
        print("  [SKIP] pycox not available")
        return None

    set_seed(seed)
    train_df = _sort_deterministic(train_df)

    X_train, remaining_time, events = prepare_remaining_time_data(train_df, feature_cols)
    if len(X_train) == 0:
        print("  [ERROR] No valid training data")
        return None

    X_clean, scaler, valid_features = preprocess_features(X_train, fit_scaler=True)
    if X_clean.shape[1] == 0:
        print("  [ERROR] No valid features after preprocessing")
        return None

    in_features = X_clean.shape[1]
    labtrans = LogisticHazard.label_transform(num_durations)
    y_train = labtrans.fit_transform(remaining_time.astype('float64'), events.astype('float64'))

    print(f"  Training LogHaz on {len(X_clean)} samples, {in_features} features, "
          f"{num_durations} duration bins...")
    net = tt.practical.MLPVanilla(in_features, [128, 64], labtrans.out_features,
                                  batch_norm=True, dropout=0.1)
    model = LogisticHazard(net, tt.optim.Adam, duration_index=labtrans.cuts)
    model.optimizer.set_lr(lr)
    model.fit(X_clean.astype('float32'), y_train, batch_size=batch_size,
              epochs=num_epochs, verbose=False)

    model.scaler_ = scaler
    model.feature_mask_ = valid_features
    model.feature_cols_ = feature_cols
    return model


# =============================================================================
# Classification / regression baselines
# =============================================================================

def train_logreg(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    C: float = 1.0,
    seed: int = 42,
) -> Any:
    """
    Logistic regression on binary winner label (scikit-learn).

    Hyperparameter search grid used for the paper:
        C (inverse regularization): {0.01, 0.1, 1.0, 10.0}
    """
    from sklearn.linear_model import LogisticRegression

    set_seed(seed)
    train_df = _sort_deterministic(train_df)

    cols = _classifier_feature_cols(feature_cols, train_df)
    X = np.nan_to_num(train_df[cols].values.astype(float), nan=0.0)
    y = train_df['squad_win'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Training LogReg on {len(X)} samples, {X.shape[1]} features...")
    model = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1, C=C)
    model.fit(X_scaled, y)
    model.scaler_ = scaler
    model.feature_cols_ = cols

    original_predict_proba = model.predict_proba

    def scaled_predict_proba(X):
        return original_predict_proba(model.scaler_.transform(X))

    model.predict_proba = scaled_predict_proba
    return model


def train_rf_classifier(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    n_estimators: int = 100,
    max_depth: int = 10,
    seed: int = 42,
) -> Any:
    """
    Random forest classifier on binary winner label (scikit-learn).

    Hyperparameter search grid used for the paper:
        n_estimators: {100, 300}, max_depth: {10, 20, None}
    """
    from sklearn.ensemble import RandomForestClassifier

    set_seed(seed)
    train_df = _sort_deterministic(train_df)

    cols = _classifier_feature_cols(feature_cols, train_df)
    X = np.nan_to_num(train_df[cols].values.astype(float), nan=0.0)
    y = train_df['squad_win'].values

    print(f"  Training RFClf on {len(X)} samples, {X.shape[1]} features...")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=seed, n_jobs=8)
    model.fit(X, y)
    model.feature_cols_ = cols
    return model


def train_lgbm_regression(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    num_leaves: int = 31,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    seed: int = 42,
) -> Optional[Any]:
    """
    LightGBM regressor predicting squad_death_time directly.

    Hyperparameter search grid used for the paper:
        num_leaves: {31, 63, 127}, learning_rate: {0.01, 0.05, 0.1},
        n_estimators: {100, 500, 1000}
    """
    if not HAS_LGBM:
        print("  [SKIP] lightgbm not available")
        return None

    set_seed(seed)
    train_df = _sort_deterministic(train_df)

    cols = _classifier_feature_cols(feature_cols, train_df)
    X = train_df[cols].fillna(0)
    y = train_df['squad_death_time']

    print(f"  Training LGBMReg on {len(X)} samples, {X.shape[1]} features...")
    model = lgb.LGBMRegressor(
        objective='regression', metric='rmse', boosting_type='gbdt',
        num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators,
        random_state=seed, verbose=-1,
    )
    model.fit(X, y)
    model.feature_cols_ = cols
    return model


def train_lgbm_classifier(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    num_leaves: int = 31,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    seed: int = 42,
) -> Optional[Any]:
    """
    LightGBM binary winner classifier.

    Hyperparameter search grid used for the paper:
        num_leaves: {31, 63, 127}, learning_rate: {0.01, 0.05, 0.1},
        n_estimators: {100, 500, 1000}
    """
    if not HAS_LGBM:
        print("  [SKIP] lightgbm not available")
        return None

    set_seed(seed)
    train_df = _sort_deterministic(train_df)

    cols = _classifier_feature_cols(feature_cols, train_df)
    X = train_df[cols].fillna(0)
    y = train_df['squad_win']

    print(f"  Training LGBMClf on {len(X)} samples, {X.shape[1]} features...")
    model = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss', boosting_type='gbdt',
        num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators,
        random_state=seed, verbose=-1,
    )
    model.fit(X, y)
    model.feature_cols_ = cols
    return model


# =============================================================================
# Rule-based baseline
# =============================================================================

def rule_based_probs(group_df: pd.DataFrame, alive_mask: np.ndarray) -> np.ndarray:
    """
    Rule-based baseline from the paper: surviving members + total squad health,
    softmax-normalized over alive squads.
    """
    values = (group_df['squad_alive_count'].values.astype(float)
              + group_df['squad_total_health'].values.astype(float))
    values = np.where(np.isnan(values) | ~alive_mask, -1e10, values)
    probs = softmax(values)
    probs = np.where(alive_mask, probs, 0.0)
    alive_sum = probs[alive_mask].sum()
    if alive_sum > 0:
        probs[alive_mask] = probs[alive_mask] / alive_sum
    return probs


# =============================================================================
# Unified prediction interface
# =============================================================================

def predict_winner_probs(
    model: Any,
    model_name: str,
    X_group: np.ndarray,
    alive_mask: np.ndarray,
    time_point: Optional[float] = None,
    match_id: Optional[str] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Predict winner probabilities for all squads at a single (match, time_point).

    Args:
        model: Trained model (None for RandomGuess).
        model_name: One of 'RandomGuess', 'CoxPH', 'DeepSurv', 'LogHaz',
            'LogReg', 'RFClf', 'LGBMReg', 'LGBMClf'.
        X_group: (num_squads, num_features) raw features for all squads.
        alive_mask: (num_squads,) boolean mask for alive squads.
        time_point / match_id: used to derive a deterministic per-time-point
            seed for RandomGuess.
        seed: Base random seed.

    Returns:
        (num_squads,) probability distribution (sums to 1 over alive squads).
    """
    num_squads = len(X_group)
    probs = np.zeros(num_squads)

    if alive_mask.sum() == 0:
        return probs

    alive_indices = np.where(alive_mask)[0]

    if model_name == 'RandomGuess':
        # Deterministic per-(match, time_point) seed
        match_hash = hash(str(match_id)) % (2 ** 32) if match_id is not None else 0
        time_hash = hash(float(time_point)) % (2 ** 32) if time_point is not None else 0
        rng = np.random.RandomState((seed + match_hash + time_hash) % (2 ** 32))
        selected = alive_indices[rng.choice(len(alive_indices))]
        probs[selected] = 1.0
        return probs

    if model_name in ('CoxPH', 'DeepSurv'):
        X_alive = X_group[alive_indices]
        X_alive = model.scaler_.transform(X_alive)[:, model.feature_mask_]
        if model_name == 'CoxPH':
            X_alive = np.clip(X_alive, -3.0, 3.0)
            scores = -model.predict(X_alive)
        else:
            log_hazard = model.predict(X_alive.astype('float32'))
            if isinstance(log_hazard, torch.Tensor):
                log_hazard = log_hazard.cpu().numpy()
            scores = -np.array(log_hazard).flatten()
        probs[alive_indices] = softmax(np.array(scores).flatten())
        return probs

    if model_name == 'LogHaz':
        X_alive = X_group[alive_indices]
        X_alive = model.scaler_.transform(X_alive)[:, model.feature_mask_]
        # Mean survival probability across duration bins as score
        surv = model.predict_surv_df(X_alive.astype('float32'))
        scores = surv.mean(axis=0).values.flatten()
        probs[alive_indices] = softmax(scores)
        return probs

    if model_name == 'LGBMReg':
        scores = np.array(model.predict(X_group[alive_indices])).flatten()
        probs[alive_indices] = softmax(scores)
        return probs

    if model_name in ('LogReg', 'RFClf', 'LGBMClf'):
        pred_proba = model.predict_proba(X_group)
        raw = pred_proba[:, 1] if pred_proba.ndim == 2 else pred_proba.ravel()
        raw = np.array(raw).flatten()
        raw[~alive_mask] = 0.0
        alive_sum = raw[alive_mask].sum()
        if alive_sum > 0:
            raw[alive_mask] /= alive_sum
        else:
            raw[alive_mask] = 1.0 / alive_mask.sum()
        return raw

    raise ValueError(f"Unknown model_name: {model_name}")
