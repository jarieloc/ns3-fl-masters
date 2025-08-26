
# flsim/fl_model.py (MedBIoT variant)
# Drop this file into your flsim/ as fl_model.py when running MedBIoT.
# It mirrors the API/structure of your NB-AIoT fl_model.py but:
#   - Looks for data under <config.paths.data>/medbiot/**.csv
#   - If not found, auto-downloads a small benign/malware pair from TalTech
#   - Builds a numeric-only tabular dataset, standardizes features, and returns TensorDatasets

import logging
from typing import Optional, List, Tuple
from pathlib import Path
import glob
import os
import shutil

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

import load_data  # base class from this repo

# -------------------- Data loader knobs --------------------
# Resolved relative to config.paths.data (e.g., <ROOT>/data/medbiot/**/*.csv)
CSV_GLOB = "medbiot/**/*.csv"

# Modest defaults so pipeline runs even on limited RAM; tune per node
MAX_FILES        = int(os.environ.get("MEDBIOT_MAX_FILES", "4"))
NROWS_PER_FILE   = int(os.environ.get("MEDBIOT_NROWS_PER_FILE", "200000"))
CHUNKSIZE        = int(os.environ.get("MEDBIOT_CHUNKSIZE", "100000"))
TEST_FRACTION    = float(os.environ.get("MEDBIOT_TEST_FRACTION", "0.30"))
RANDOM_SEED      = int(os.environ.get("MEDBIOT_SEED", "42"))

# Download a smaller pair by default (reasonable size, good signal)
# You can switch to the larger pair by setting MEDBIOT_PAIR=large
PAIR = os.environ.get("MEDBIOT_PAIR", "small").lower()
SMALL_MAL_URL = "https://cs.taltech.ee/research/data/medbiot/bulk/structured_dataset/malware/torii_mal_all.csv"
SMALL_BEN_URL = "https://cs.taltech.ee/research/data/medbiot/fine-grained/structured_dataset/normal/mirai_leg_fan.csv"
LARGE_MAL_URL = "https://cs.taltech.ee/research/data/medbiot/bulk/structured_dataset/malware/mirai_mal_CC_all.csv"
LARGE_BEN_URL = "https://cs.taltech.ee/research/data/medbiot/fine-grained/structured_dataset/normal/mirai_leg.csv"

# Training settings (kept in parity with your NB-AIoT module)
lr = 1e-3
log_interval = 10
rou = 1
loss_thres = 0.01

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# Will be set after reading data
_INPUT_DIM: Optional[int] = None


def _label_from_path(p: Path) -> int:
    s = p.as_posix().lower()
    # normal/leg => 0 ; mal/malware => 1
    if "normal" in s or "leg" in s:
        return 0
    return 1


def _read_numeric_limited(path: Path, nrows: int, chunksize: int) -> pd.DataFrame:
    """Read up to nrows numeric rows from CSV via chunking (robust to huge files)."""
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
        # last resort: simple small read
        num = pd.read_csv(path, nrows=nrows, low_memory=False).select_dtypes(include=[np.number])
        if num.empty:
            raise ValueError(f"No numeric columns detected in {path}")
        frames = [num]
    df = pd.concat(frames, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)
    return df


def _build_from_csvs(files: List[Path]) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Build (X_df, y, feature_names) using only *common* numeric columns across selected files."""
    if MAX_FILES and MAX_FILES > 0:
        files = files[:MAX_FILES]

    logging.info(f"[MedBIoT] Using {len(files)} CSVs (nrows_per_file={NROWS_PER_FILE}, chunksize={CHUNKSIZE})")
    # First pass: find common numeric columns via peeks
    numeric_sets = []
    for p in files:
        try:
            head = pd.read_csv(p, nrows=2000, low_memory=False)
            numeric_sets.append(set(head.select_dtypes(include=[np.number]).columns.tolist()))
        except Exception as e:
            logging.warning(f"[MedBIoT] Peek failed for {p.name}: {e}")
    common_cols = set.intersection(*numeric_sets) if numeric_sets else set()
    if not common_cols:
        raise RuntimeError("No common numeric feature columns across MedBIoT files.")

    frames: List[pd.DataFrame] = []
    labels: List[np.ndarray] = []

    for p in files:
        try:
            df = _read_numeric_limited(p, NROWS_PER_FILE, CHUNKSIZE)
            Xdf = df.loc[:, sorted(list(common_cols))].copy()
            y = np.full(len(Xdf), _label_from_path(p), dtype=np.int64)
            frames.append(Xdf)
            labels.append(y)
            logging.info(f"[MedBIoT] {p.name}: used {len(Xdf)} rows, label={int(y[0])}")
        except Exception as e:
            logging.warning(f"[MedBIoT] Skipping {p.name}: {e}")

    if not frames:
        raise RuntimeError("No data loaded from CSVs (all skipped).")

    X_df = pd.concat(frames, ignore_index=True)
    y = np.concatenate(labels)
    feat_names = sorted(list(common_cols))
    return X_df, y, feat_names


def _build_synthetic(n_samples=6000, n_features=23, anomaly_fraction=0.2):
    rng = np.random.default_rng(42)
    n_anom = int(n_samples * anomaly_fraction)
    n_norm = n_samples - n_anom
    X_norm = rng.normal(size=(n_norm, n_features))
    X_anom = rng.normal(loc=3.0, scale=1.5, size=(n_anom, n_features))
    X = np.vstack([X_norm, X_anom])
    y = np.hstack([np.zeros(n_norm, dtype=int), np.ones(n_anom, dtype=int)])
    perm = rng.permutation(len(y))
    X = X[perm]; y = y[perm]
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), y, [f"f{i}" for i in range(n_features)]


def _download(url: str, dst: Path) -> bool:
    """Stream-download a file; fallback to wget if requests/urllib are unavailable."""
    try:
        # Try requests first
        import requests  # type: ignore
        dst.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
        logging.info(f"[MedBIoT] Downloaded: {url} -> {dst}")
        return True
    except Exception as e:
        logging.warning(f"[MedBIoT] requests failed ({e}); trying urllib/wget...")
        try:
            import urllib.request
            dst.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dst.as_posix())
            logging.info(f"[MedBIoT] Downloaded via urllib: {url} -> {dst}")
            return True
        except Exception as e2:
            # Last resort: system wget if available
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                rc = os.system(f"wget -q {url} -O {dst.as_posix()}")
                if rc == 0 and dst.exists() and dst.stat().st_size > 0:
                    logging.info(f"[MedBIoT] Downloaded via wget: {url} -> {dst}")
                    return True
            except Exception as e3:
                pass
            logging.error(f"[MedBIoT] Failed to download {url}: {e2}")
            return False


def _ensure_medbiot_local(data_root: Path) -> List[Path]:
    """
    Ensure we have MedBIoT CSVs locally.
    Returns a list of CSV paths under data_root/medbiot/...
    Strategy:
      1) If CSVs already exist under medbiot/** -> use them.
      2) Otherwise, download a small benign/malware pair into the expected structure.
      3) If downloads fail, return empty list.
    """
    files = [Path(p) for p in glob.glob(str(data_root / CSV_GLOB))]
    if files:
        return sorted(files)

    # Build expected dirs
    mal_dir = data_root / "medbiot" / "structured_dataset" / "malware"
    ben_dir = data_root / "medbiot" / "structured_dataset" / "normal"
    mal_dir.mkdir(parents=True, exist_ok=True)
    ben_dir.mkdir(parents=True, exist_ok=True)

    if PAIR == "large":
        mal_url, ben_url = LARGE_MAL_URL, LARGE_BEN_URL
        mal_name, ben_name = "mirai_mal_CC_all.csv", "mirai_leg.csv"
    else:
        mal_url, ben_url = SMALL_MAL_URL, SMALL_BEN_URL
        mal_name, ben_name = "torii_mal_all.csv", "mirai_leg_fan.csv"

    mal_dst = mal_dir / mal_name
    ben_dst = ben_dir / ben_name

    ok_mal = mal_dst.exists() and mal_dst.stat().st_size > 0
    ok_ben = ben_dst.exists() and ben_dst.stat().st_size > 0
    if not ok_mal:
        ok_mal = _download(mal_url, mal_dst)
    if not ok_ben:
        ok_ben = _download(ben_url, ben_dst)

    if ok_mal and ok_ben:
        files = [Path(p) for p in glob.glob(str(data_root / CSV_GLOB))]
        return sorted(files)
    else:
        logging.error("[MedBIoT] Could not ensure local CSVs. Falling back to synthetic.")
        return []


class Generator(load_data.Generator):
    """Generator for MedBIoT: auto-download from TalTech or synthetic fallback."""

    def read(self, path):
        global _INPUT_DIM

        rng = np.random.default_rng(RANDOM_SEED)
        root = Path(path)

        # 1) Discover CSVs locally
        files = [Path(p) for p in glob.glob(str(root / CSV_GLOB))]

        # 2) If none, try to download into <root>/medbiot/structured_dataset/{malware,normal}/
        if not files:
            files = _ensure_medbiot_local(root)

        # 3) If still none, synthetic fallback
        used_synth = False
        if not files:
            X_df, y, feat_names = _build_synthetic()
            used_synth = True
        else:
            X_df, y, feat_names = _build_from_csvs(files)

        # Shuffle once (global)
        perm = rng.permutation(len(y))
        X_df = X_df.iloc[perm].reset_index(drop=True)
        y = y[perm]

        # Standardize columns
        mu = X_df.mean(axis=0)
        sigma = X_df.std(axis=0).replace(0, 1.0)
        X_df = (X_df - mu) / sigma
        X = torch.tensor(X_df.values, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        _INPUT_DIM = X.shape[1]

        if used_synth:
            logging.info(f"[MedBIoT] (synthetic) features={_INPUT_DIM}, total={len(y)}, "
                         f"benign={(y==0).sum()}, mal={(y==1).sum()}")
        else:
            logging.info(f"[MedBIoT] features={_INPUT_DIM}, total={len(y)}, "
                         f"benign={(y==0).sum()}, mal={(y==1).sum()}")

        # Split into train/test
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
        self.labels = ["benign", "malicious"]


class SimpleClassifier(nn.Module):
    """MLP: input -> 64 -> 32 -> 2 (matches your notebook's spirit)."""
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


# Framework hook: Net() must be constructible without args
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
            weights.append((name, p.detach().clone().to("cpu")))
    return weights


def load_weights(model, weights):
    updated = {name: w for name, w in weights}
    model.load_state_dict(updated, strict=False)


def flatten_weights(weights):
    vecs = []
    for _, w in weights:
        vecs.append(w.detach().view(-1).cpu().numpy())
    return np.concatenate(vecs) if vecs else np.array([], dtype=np.float32)


def train(model, trainloader, optimizer, epochs, reg=None, dp: Optional[dict] = None):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)

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

    if reg is not None:
        old_weights = flatten_weights(extract_weights(model))
        old_weights = torch.from_numpy(old_weights)

    last_loss = None
    for epoch in range(1, epochs + 1):
        for batch_id, (xb, yb) in enumerate(trainloader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

            if reg is not None:
                new_weights = flatten_weights(extract_weights(model))
                new_weights = torch.from_numpy(new_weights)
                mse_loss = nn.MSELoss(reduction="sum")
                l2_loss = rou / 2 * mse_loss(new_weights, old_weights)
                l2_loss = l2_loss.to(torch.float32)
                loss = loss + l2_loss

            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            if batch_id % log_interval == 0:
                logging.debug(f"Epoch [{epoch}/{epochs}] Loss: {last_loss:.6f}")

            if last_loss < loss_thres:
                if privacy_engine is not None:
                    try:
                        delta = float(dp.get("delta", 1e-5))
                        eps = (privacy_engine.accountant.get_epsilon(delta)
                               if hasattr(privacy_engine, "accountant")
                               else privacy_engine.get_epsilon(delta))
                        logging.info(f"[DP] early-exit: ε≈{eps:.2f}, δ={delta}")
                    except Exception as e:
                        logging.debug(f"[DP] epsilon unavailable: {e}")
                return last_loss

        # Per-epoch DP log
        if privacy_engine is not None:
            try:
                delta = float(dp.get("delta", 1e-5))
                eps = (privacy_engine.accountant.get_epsilon(delta)
                       if hasattr(privacy_engine, "accountant")
                       else privacy_engine.get_epsilon(delta))
                logging.info(f"[DP] epoch {epoch}: ε≈{eps:.2f}, δ={delta}")
            except Exception as e:
                logging.debug(f"[DP] epsilon unavailable: {e}")

    if last_loss is not None:
        logging.info("loss: {:.6f}".format(last_loss))
    return last_loss if last_loss is not None else 0.0


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
    logging.debug("Accuracy: {:.2f}%".format(100 * accuracy))
    return accuracy
