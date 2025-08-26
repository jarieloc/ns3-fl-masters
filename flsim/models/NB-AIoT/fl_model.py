# flsim/fl_model.py
import logging
from typing import Optional, List
from pathlib import Path
import glob
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F  # not used in MLP forward but kept for parity
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

import load_data  # base class from this repo

# -------------------- Data loader knobs (adjust as needed) --------------------
CSV_GLOB = "nbaiot/*.csv"    # resolved relative to config.paths.data
MAX_FILES = 6                # like your notebook
NROWS_PER_FILE = 50_000
CHUNKSIZE = 100_000
TEST_FRACTION = 0.30         # 70/30 train/test
RANDOM_SEED = 42
# -----------------------------------------------------------------------------

# Training settings (match your notebook-ish defaults)
lr = 1e-3                    # Adam LR
log_interval = 10
rou = 1
loss_thres = 0.01            # early-stop threshold (kept from original repo)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# Will be set by Generator.read() once we know feature dimension
_INPUT_DIM: Optional[int] = None


def _label_from_filename(p: Path) -> int:
    s = p.as_posix().lower()
    # benign/normal => 0, everything else => 1
    return 0 if ('benign' in s or 'normal' in s) else 1


def _read_limited_rows(path: Path, nrows: int, chunksize: int) -> pd.DataFrame:
    rows_left = nrows
    frames: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        num = chunk.select_dtypes(include=[np.number])
        if num.empty:
            continue
        if len(num) > rows_left:
            frames.append(num.iloc[:rows_left].copy())
            rows_left = 0
            break
        frames.append(num)
        rows_left -= len(num)
        if rows_left <= 0:
            break
    if not frames:
        # try a simple small read as fallback
        num = pd.read_csv(path, nrows=nrows, low_memory=False).select_dtypes(include=[np.number])
        if num.empty:
            raise ValueError(f"No numeric columns detected in {path}")
        frames = [num]
    return pd.concat(frames, ignore_index=True)


def _try_kagglehub_download() -> Optional[Path]:
    """
    Try to fetch the N-BaIoT dataset via kagglehub.
    Returns the root path of the cached dataset if successful, else None.
    """
    try:
        import kagglehub
        logging.info("[NB-AIoT] Attempting Kagglehub download: mkashifn/nbaiot-dataset ...")
        root = Path(kagglehub.dataset_download("mkashifn/nbaiot-dataset"))
        # sanity check: must contain at least one CSV
        if list(root.rglob("*.csv")):
            logging.info(f"[NB-AIoT] Kagglehub cache found at: {root}")
            return root
        logging.warning("[NB-AIoT] Kagglehub path has no CSVs after download.")
        return None
    except Exception as e:
        logging.warning(f"[NB-AIoT] Kagglehub unavailable or failed: {e}")
        return None


def _build_from_csvs(files: List[Path]) -> tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Build a combined numeric dataset (X_df, y, feature_names) from a list of CSV files.
    Uses up to MAX_FILES and NROWS_PER_FILE per file. Keeps only common numeric columns.
    """
    if MAX_FILES and MAX_FILES > 0:
        files = files[:MAX_FILES]

    logging.info(f"[NB-AIoT] Loading {len(files)} CSVs "
                 f"(nrows_per_file={NROWS_PER_FILE}, chunksize={CHUNKSIZE})")

    # First pass: find common numeric columns via a small peek
    numeric_sets = []
    for p in files:
        try:
            head = pd.read_csv(p, nrows=1000, low_memory=False)
            numeric_sets.append(set(head.select_dtypes(include=[np.number]).columns.tolist()))
        except Exception as e:
            logging.warning(f"[NB-AIoT] Peek failed for {p.name}: {e}")
    common_cols = set.intersection(*numeric_sets) if numeric_sets else set()
    if not common_cols:
        raise RuntimeError("No common numeric feature columns across files.")

    frames: List[pd.DataFrame] = []
    labels: List[np.ndarray] = []

    for p in files:
        try:
            df = _read_limited_rows(p, NROWS_PER_FILE, CHUNKSIZE)
            Xdf = df.loc[:, sorted(list(common_cols))].copy()
            Xdf.replace([np.inf, -np.inf], np.nan, inplace=True)
            Xdf.fillna(0.0, inplace=True)
            y = np.full(len(Xdf), _label_from_filename(p), dtype=np.int64)

            frames.append(Xdf)
            labels.append(y)
            logging.info(f"[NB-AIoT] {p.name}: used {len(Xdf)} rows, label={int(y[0])}")
        except Exception as e:
            logging.warning(f"[NB-AIoT] Skipping {p.name}: {e}")

    if not frames:
        raise RuntimeError("No data loaded from CSVs (all skipped).")

    X_df = pd.concat(frames, ignore_index=True)
    y = np.concatenate(labels)
    feature_names = sorted(list(common_cols))
    return X_df, y, feature_names


def _build_synthetic(num_samples=6000, num_features=115, anomaly_fraction=0.2):
    rng = np.random.default_rng(42)
    n_anom = int(num_samples * anomaly_fraction)
    n_norm = num_samples - n_anom
    X_norm = rng.normal(loc=0.0, scale=1.0, size=(n_norm, num_features))
    X_anom = rng.normal(loc=3.0, scale=1.5, size=(n_anom, num_features))
    X_df = pd.DataFrame(np.vstack([X_norm, X_anom]),
                        columns=[f"f{i}" for i in range(num_features)])
    y = np.concatenate([np.zeros(n_norm, dtype=int), np.ones(n_anom, dtype=int)])
    perm = rng.permutation(len(y))
    return X_df.iloc[perm].reset_index(drop=True), y[perm], list(X_df.columns)


class Generator(load_data.Generator):
    """Generator for NB-AIoT: auto-download (kagglehub) or synthetic fallback."""

    def read(self, path):
        global _INPUT_DIM

        rng = np.random.default_rng(RANDOM_SEED)
        root = Path(path)

        # 1) Look for local CSVs under <data>/nbaiot/*.csv
        files = sorted(Path(p) for p in glob.glob(str(root / CSV_GLOB)))

        # 2) If none, try kagglehub cache (read directly from cache to avoid copying huge files)
        if not files:
            kh_root = _try_kagglehub_download()
            if kh_root is not None:
                files = sorted(kh_root.rglob("*.csv"))

        # 3) If still none, synthetic fallback so the pipeline always runs
        used_synthetic = False
        if not files:
            logging.warning("[NB-AIoT] No CSVs available locally or via Kagglehub. Using synthetic fallback.")
            X_df, y, feat_names = _build_synthetic()
            used_synthetic = True
        else:
            X_df, y, feat_names = _build_from_csvs(files)

        # Shuffle once (like your notebook)
        perm = rng.permutation(len(y))
        X_df = X_df.iloc[perm].reset_index(drop=True)
        y = y[perm]

        # Standardize (fit on full data once)
        mu = X_df.mean(axis=0)
        sigma = X_df.std(axis=0).replace(0, 1.0)
        X_df = (X_df - mu) / sigma

        X = torch.tensor(X_df.values, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        _INPUT_DIM = X.shape[1]
        if used_synthetic:
            logging.info(f"[NB-AIoT] (synthetic) features={_INPUT_DIM}, total={len(y)}, "
                         f"benign={(y==0).sum()}, anomaly={(y==1).sum()}")
        else:
            logging.info(f"[NB-AIoT] features={_INPUT_DIM}, total={len(y)}, "
                         f"benign={(y==0).sum()}, anomaly={(y==1).sum()}")

        # Split train/test (70/30)
        n_total = len(y)
        n_test = int(n_total * TEST_FRACTION)
        n_train = n_total - n_test
        ds = TensorDataset(X, y_t)
        trainset, testset = random_split(
            ds, [n_train, n_test],
            generator=torch.Generator().manual_seed(RANDOM_SEED)
        )

        self.trainset = trainset
        self.testset = testset
        self.labels = ["benign", "anomaly"]


class SimpleClassifier(nn.Module):
    """MLP: input -> 64 -> 32 -> 2"""
    def __init__(self, input_dim: Optional[int] = None, hidden1=64, hidden2=32, num_classes=2):
        super().__init__()
        d = input_dim if input_dim is not None else (_INPUT_DIM or 1)
        self.net = nn.Sequential(
            nn.Linear(d, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # logits


# The framework expects Net() with no args
class Net(SimpleClassifier):
    def __init__(self):
        super().__init__(input_dim=_INPUT_DIM)


def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=lr)


def get_trainloader(trainset, batch_size):
    return DataLoader(trainset, batch_size=batch_size, shuffle=True)


def get_testloader(testset, batch_size):
    return DataLoader(testset, batch_size=batch_size, shuffle=False)


def extract_weights(model):
    weights = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            weights.append((name, p.detach().clone().to('cpu')))
    return weights


def load_weights(model, weights):
    updated = {name: w for name, w in weights}
    model.load_state_dict(updated, strict=False)


def flatten_weights(weights):
    vecs = []
    for _, w in weights:
        vecs.append(w.detach().view(-1).cpu().numpy())
    return np.concatenate(vecs)


def train(model, trainloader, optimizer, epochs, reg=None, dp: Optional[dict] = None):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)

    # --- Opacus DP (optional) ---
    privacy_engine = None
    if dp and dp.get("enable", False):
        try:
            from opacus import PrivacyEngine
            noise_multiplier = float(dp.get("noise_multiplier", 0.8))
            max_grad_norm = float(dp.get("max_grad_norm", 1.0))
            secure_mode = bool(dp.get("secure_mode", False))
            accountant = dp.get("accountant", "rdp")

            privacy_engine = PrivacyEngine(accountant=accountant, secure_mode=secure_mode)
            model, optimizer, trainloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=trainloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
            logging.info(f"[DP] Enabled: sigma={noise_multiplier}, C={max_grad_norm}, accountant={accountant}")
        except Exception as e:
            logging.warning(f"[DP] Failed to enable DP-SGD ({e}). Continuing without DP.")
            privacy_engine = None

    # --- Regularization snapshot (as in original code path) ---
    if reg is not None:
        old_weights = flatten_weights(extract_weights(model))
        old_weights = torch.from_numpy(old_weights)

    for epoch in range(1, epochs + 1):
        for batch_id, (xb, yb) in enumerate(trainloader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

            if reg is not None:
                new_weights = flatten_weights(extract_weights(model))
                new_weights = torch.from_numpy(new_weights)
                mse_loss = nn.MSELoss(reduction='sum')
                l2_loss = rou / 2 * mse_loss(new_weights, old_weights)
                l2_loss = l2_loss.to(torch.float32)
                loss = loss + l2_loss

            loss.backward()
            optimizer.step()

            if batch_id % log_interval == 0:
                logging.debug(f'Epoch: [{epoch}/{epochs}]\tLoss: {loss.item():.6f}')

            if loss.item() < loss_thres:
                if privacy_engine is not None:
                    try:
                        delta = float(dp.get("delta", 1e-5))
                        eps = (privacy_engine.accountant.get_epsilon(delta)
                               if hasattr(privacy_engine, "accountant")
                               else privacy_engine.get_epsilon(delta))
                        logging.info(f"[DP] early-exit: ε≈{eps:.2f}, δ={delta}, "
                                     f"σ={dp.get('noise_multiplier', 0.8)}, C={dp.get('max_grad_norm', 1.0)}")
                    except Exception as e:
                        logging.debug(f"[DP] epsilon unavailable: {e}")
                return loss.item()

        # end of epoch DP log
        if privacy_engine is not None:
            try:
                delta = float(dp.get("delta", 1e-5))
                eps = (privacy_engine.accountant.get_epsilon(delta)
                       if hasattr(privacy_engine, "accountant")
                       else privacy_engine.get_epsilon(delta))
                logging.info(f"[DP] epoch {epoch}: ε≈{eps:.2f}, δ={delta}, "
                             f"σ={dp.get('noise_multiplier', 0.8)}, C={dp.get('max_grad_norm', 1.0)}")
            except Exception as e:
                logging.debug(f"[DP] epsilon unavailable: {e}")

    logging.info('loss: {:.6f}'.format(loss.item()))
    if privacy_engine is not None:
        try:
            delta = float(dp.get("delta", 1e-5))
            eps = (privacy_engine.accountant.get_epsilon(delta)
                   if hasattr(privacy_engine, "accountant")
                   else privacy_engine.get_epsilon(delta))
            logging.info(f"[DP] final: ε≈{eps:.2f}, δ={delta}, "
                         f"σ={dp.get('noise_multiplier', 0.8)}, C={dp.get('max_grad_norm', 1.0)}")
        except Exception as e:
            logging.debug(f"[DP] epsilon unavailable: {e}")
    return loss.item()


@torch.no_grad()
def test(model, testloader):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)

    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in testloader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * yb.size(0)
        pred = logits.argmax(dim=1, keepdim=False)
        correct += (pred == yb).sum().item()
        total += yb.size(0)

    accuracy = correct / max(total, 1)
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))
    return accuracy
