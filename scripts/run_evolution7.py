import csv
import gc
import hashlib
import json
import math
import os
import pickle
import random
import socket
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings(
    "ignore",
    message=r".*Your input ran out of data; interrupting training.*",
    category=UserWarning,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from deap import base, creator, tools
from sklearn.metrics import precision_recall_fscore_support
from tensorflow import keras
from tensorflow.keras import layers

CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("GA_DATA_ROOT", Path.cwd())).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("GA_OUTPUT_ROOT", Path.cwd())).expanduser().resolve()

TRAIN_CACHE_DIR = DATA_ROOT / "data/0.25/train"
TEST_AND_VAL_CACHE_DIR = DATA_ROOT / "data/0.25/test_and_val"
TARG_DEDUPED_CSV = DATA_ROOT / "data/encoded/targ_deduped.csv"
FEATURE_MANIFEST_CSV = DATA_ROOT / "data/0.25/feature_manifest.csv"

OUT_DIR = OUTPUT_ROOT / "ga_results"
POP_DIR = OUT_DIR / "population_files"
REPORT_DIR = OUT_DIR / "ga_progress_reports"
INDIV_DIR = OUT_DIR / "individuals_results"

POPULATION_SIZE = 100
MIN_FEATS = 30
MAX_FEATS = 597
NUM_GENERATIONS = 50
NUM_GEN_REPLACE = 10
BOTTOM_FRAC_REPLACE = 0.1
STARTING_BITS_FLIPPED = 5
DECAY_BIT_FLIPS_EVERY = 10
REDUCE_BIT_FLIPS_BY = 1
GA_TRAIN_FRAC = 0.25
GA_VAL_FRAC = 0.15
FINAL_VAL_FRAC = 0.35
MAX_WORKERS = int(os.environ.get("GA_MAX_WORKERS", "1"))
REFERENCE_POINTS = 91

GA_LSTM_EPOCHS = 15
GA_LSTM_EARLY_STOPPING = 5
FINAL_LSTM_EPOCHS = 250
FINAL_LSTM_EARLY_STOPPING = 10

PAD_X_VALUE = -999.0
BATCH_SIZE = 50
LR = 1e-4
LSTM_UNITS = 100
SPATIAL_DROPOUT = 0.1
LSTM_DROPOUT = 0.1
RECURRENT_DROPOUT = 0.0
USE_CB_WEIGHTS = True
CB_BETAS = np.array([0.99999, 0.99999], dtype=np.float32)
NEGATIVE_RESIDUE_WEIGHT = 0.5
TERMINAL_POSITIVE_WEIGHT = 0.25
INTERNAL_POSITIVE_WEIGHT = 1.0
F1_MODE = "mean_pr"
PRED_THRESHOLD = 0.55
FINAL_EVAL_THRESHOLDS = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
RANDOM_SEED = 42
PREFETCH_N = 2
TEST_HEATMAP_N = 100
TEST_HEATMAP_FILE = "heatmap_test100_pred_before_after_true_triptych.png"
FULL_VAL_SEQUENCE_PRED_CSV = "full_val_sequence_predictions.csv"
DOMAIN_COUNT_BARPLOT_BEFORE_FILE = "barplot_test_domain_count_categories_before_smoothing.png"
DOMAIN_COUNT_BARPLOT_AFTER_FILE = "barplot_test_domain_count_categories_after_smoothing.png"
min_residues_1s_smooth = 9
num_residues_0_domain_definition = 10

FORCE_SERIAL_ON_GPU = os.environ.get("GA_FORCE_SERIAL_ON_GPU", "1") == "1"
SYNC_COMMAND = os.environ.get("GA_SYNC_COMMAND", "").strip()
SYNC_EVERY_GENERATION = os.environ.get("GA_SYNC_EVERY_GENERATION", "1") == "1"
SYNC_EVERY_INDIVIDUAL = os.environ.get("GA_SYNC_EVERY_INDIVIDUAL", "1") == "1"

STATE_PATH = POP_DIR / "current_state.json"
CONFIG_PATH = POP_DIR / "ga_config_snapshot.json"
SPLITS_PATH = POP_DIR / "data_splits.json"
RNG_PATH = POP_DIR / "rng_state.pkl"
CACHE_PATH = POP_DIR / "fitness_cache.pkl"
FINAL_RETRAIN_STATE_PATH = POP_DIR / "final_retrain_state.json"
HISTORY_CSV_PATH = REPORT_DIR / "ga_history.csv"
LOG_PATH = REPORT_DIR / "ga_run.log"
AVG_PLOT_PATH = REPORT_DIR / "avg_metrics_and_avg_size_over_generations.png"
BEST_PLOT_PATH = REPORT_DIR / "best_metrics_over_generations.png"
INDIVIDUAL_THRESHOLD_SUMMARY_CSV = "threshold_metrics_summary.csv"
RUNTIME_INFO_PATH = REPORT_DIR / "runtime_info.json"

_WORKER_GA_TRAIN_RECORDS = None
_WORKER_GA_VAL_RECORDS = None

if not hasattr(creator, "FitnessMultiGA"):
    creator.create("FitnessMultiGA", base.Fitness, weights=(1.0, 1.0, 1.0))
if not hasattr(creator, "IndividualGA"):
    creator.create("IndividualGA", list, fitness=creator.FitnessMultiGA)


@keras.utils.register_keras_serializable(package="ga")
class ValidMaskLayer(layers.Layer):
    def call(self, inputs):
        return tf.reduce_any(tf.not_equal(inputs, PAD_X_VALUE), axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])


@keras.utils.register_keras_serializable(package="ga")
class ResidueOutLayer(layers.Layer):
    supports_masking = True

    def call(self, inputs, mask=None):
        return tf.squeeze(inputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

    def compute_mask(self, inputs, mask=None):
        return mask


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    POP_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    INDIV_DIR.mkdir(parents=True, exist_ok=True)


def log_line(msg):
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def show_progress(prefix, current, total, width=28):
    total = max(total, 1)
    frac = current / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if current >= total else "\r"
    print(f"{prefix} [{bar}] {current}/{total}", end=end, flush=True)


def save_json(path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(path, obj):
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path, default=None):
    if not path.exists():
        return default
    with path.open("rb") as f:
        return pickle.load(f)


def save_rng_state():
    save_pickle(RNG_PATH, {"python": random.getstate(), "numpy": np.random.get_state()})


def load_rng_state():
    state = load_pickle(RNG_PATH)
    if state is None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    tf.random.set_seed(RANDOM_SEED)


def configure_tensorflow_runtime():
    gpu_names = []
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
            gpu_names.append(gpu.name)
        return gpus, gpu_names
    except Exception:
        return [], []


def gpu_is_available():
    try:
        return len(tf.config.list_physical_devices("GPU")) > 0
    except Exception:
        return False


def effective_max_workers():
    requested = max(1, int(MAX_WORKERS))
    if requested > 1 and gpu_is_available() and FORCE_SERIAL_ON_GPU:
        return 1
    return requested


def runtime_info():
    _, gpu_names = configure_tensorflow_runtime()
    return {
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "code_root": str(CODE_ROOT),
        "data_root": str(DATA_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "train_cache_dir": str(TRAIN_CACHE_DIR),
        "test_and_val_cache_dir": str(TEST_AND_VAL_CACHE_DIR),
        "targ_deduped_csv": str(TARG_DEDUPED_CSV),
        "feature_manifest_csv": str(FEATURE_MANIFEST_CSV),
        "requested_max_workers": int(MAX_WORKERS),
        "effective_max_workers": int(effective_max_workers()),
        "force_serial_on_gpu": bool(FORCE_SERIAL_ON_GPU),
        "gpu_visible": bool(len(gpu_names) > 0),
        "gpu_names": gpu_names,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "sync_command_configured": bool(SYNC_COMMAND),
    }


def log_runtime_info():
    info = runtime_info()
    save_json(RUNTIME_INFO_PATH, info)
    log_line("Runtime configuration:")
    for key in sorted(info):
        log_line(f"  {key}: {info[key]}")


def maybe_run_sync_hook(reason, extra_env=None):
    if not SYNC_COMMAND:
        return
    env = os.environ.copy()
    env.update({
        "GA_SYNC_REASON": str(reason),
        "GA_DATA_ROOT": str(DATA_ROOT),
        "GA_OUTPUT_ROOT": str(OUTPUT_ROOT),
        "GA_OUT_DIR": str(OUT_DIR),
        "GA_POP_DIR": str(POP_DIR),
        "GA_REPORT_DIR": str(REPORT_DIR),
        "GA_INDIV_DIR": str(INDIV_DIR),
    })
    if extra_env:
        for key, val in extra_env.items():
            env[str(key)] = str(val)
    try:
        result = subprocess.run(
            SYNC_COMMAND,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            log_line(f"sync_hook ok | reason={reason}")
        else:
            log_line(f"sync_hook failed | reason={reason} | exit={result.returncode}")
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                log_line(f"sync_hook stdout | {line}")
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines():
                log_line(f"sync_hook stderr | {line}")
    except Exception as e:
        log_line(f"sync_hook exception | reason={reason} | {type(e).__name__}: {e}")


def make_config_snapshot():
    return {
        "CODE_ROOT": str(CODE_ROOT),
        "DATA_ROOT": str(DATA_ROOT),
        "OUTPUT_ROOT": str(OUTPUT_ROOT),
        "TRAIN_CACHE_DIR": str(TRAIN_CACHE_DIR),
        "TEST_AND_VAL_CACHE_DIR": str(TEST_AND_VAL_CACHE_DIR),
        "TARG_DEDUPED_CSV": str(TARG_DEDUPED_CSV),
        "FEATURE_MANIFEST_CSV": str(FEATURE_MANIFEST_CSV),
        "POPULATION_SIZE": POPULATION_SIZE,
        "MIN_FEATS": MIN_FEATS,
        "MAX_FEATS": MAX_FEATS,
        "NUM_GENERATIONS": NUM_GENERATIONS,
        "NUM_GEN_REPLACE": NUM_GEN_REPLACE,
        "BOTTOM_FRAC_REPLACE": BOTTOM_FRAC_REPLACE,
        "STARTING_BITS_FLIPPED": STARTING_BITS_FLIPPED,
        "DECAY_BIT_FLIPS_EVERY": DECAY_BIT_FLIPS_EVERY,
        "REDUCE_BIT_FLIPS_BY": REDUCE_BIT_FLIPS_BY,
        "GA_TRAIN_FRAC": GA_TRAIN_FRAC,
        "GA_VAL_FRAC": GA_VAL_FRAC,
        "FINAL_VAL_FRAC": FINAL_VAL_FRAC,
        "MAX_WORKERS_REQUESTED": MAX_WORKERS,
        "REFERENCE_POINTS": REFERENCE_POINTS,
        "GA_LSTM_EPOCHS": GA_LSTM_EPOCHS,
        "GA_LSTM_EARLY_STOPPING": GA_LSTM_EARLY_STOPPING,
        "FINAL_LSTM_EPOCHS": FINAL_LSTM_EPOCHS,
        "FINAL_LSTM_EARLY_STOPPING": FINAL_LSTM_EARLY_STOPPING,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": LR,
        "LSTM_UNITS": LSTM_UNITS,
        "SPATIAL_DROPOUT": SPATIAL_DROPOUT,
        "LSTM_DROPOUT": LSTM_DROPOUT,
        "RECURRENT_DROPOUT": RECURRENT_DROPOUT,
        "CB_BETAS": CB_BETAS.tolist(),
        "NEGATIVE_RESIDUE_WEIGHT": NEGATIVE_RESIDUE_WEIGHT,
        "TERMINAL_POSITIVE_WEIGHT": TERMINAL_POSITIVE_WEIGHT,
        "INTERNAL_POSITIVE_WEIGHT": INTERNAL_POSITIVE_WEIGHT,
        "PRED_THRESHOLD": PRED_THRESHOLD,
        "FINAL_EVAL_THRESHOLDS": FINAL_EVAL_THRESHOLDS,
        "RANDOM_SEED": RANDOM_SEED,
        "SMOOTH_MIN_LEN": min_residues_1s_smooth,
        "DOMAIN_ZERO_MIN_LEN": num_residues_0_domain_definition,
        "FORCE_SERIAL_ON_GPU": FORCE_SERIAL_ON_GPU,
    }


def verify_or_write_config():
    snapshot = make_config_snapshot()
    existing = load_json(CONFIG_PATH)
    if existing is None:
        save_json(CONFIG_PATH, snapshot)
        return
    if existing != snapshot:
        raise RuntimeError("Saved GA config does not match current script config. Delete ga_results or use a new output dir.")


def list_npz_files(path):
    files = sorted(path.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No NPZ files in {path}")
    return files


def inspect_feature_dim_from_first(files):
    f0 = files[0]
    with np.load(f0, allow_pickle=True) as d:
        X = d["X"]
        y = d["y"]
    if X.ndim != 2:
        raise RuntimeError(f"{f0.name}: expected X.ndim==2, got shape {X.shape}")
    if y.ndim != 1:
        raise RuntimeError(f"{f0.name}: expected y.ndim==1, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise RuntimeError(f"{f0.name}: length mismatch X[0]={X.shape[0]} vs y[0]={y.shape[0]}")
    if X.shape[0] <= 0:
        raise RuntimeError(f"{f0.name}: empty sequence is not allowed")
    return int(X.shape[1])


def load_npz_raw(path, expected_f):
    with np.load(path, allow_pickle=True) as d:
        X = d["X"].astype(np.float32, copy=False)
        y = d["y"].astype(np.float32, copy=False)
    if X.ndim != 2:
        raise RuntimeError(f"{path.name}: expected X.ndim==2, got {X.shape}")
    if y.ndim != 1:
        raise RuntimeError(f"{path.name}: expected y.ndim==1, got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise RuntimeError(f"{path.name}: X/y length mismatch")
    if X.shape[1] != expected_f:
        raise RuntimeError(f"{path.name}: expected {expected_f} features, got {X.shape[1]}")
    return X, y


def derive_internal_terminal_masks(y):
    y_bin = (y >= 0.5).astype(np.int8)
    n = int(len(y_bin))
    internal_mask = np.zeros(n, dtype=np.float32)
    terminal_mask = np.zeros(n, dtype=np.float32)
    i = 0
    while i < n:
        if y_bin[i] != 1:
            i += 1
            continue
        j = i
        while j + 1 < n and y_bin[j + 1] == 1:
            j += 1
        if i == 0 or j == n - 1:
            terminal_mask[i:j + 1] = 1.0
        else:
            internal_mask[i:j + 1] = 1.0
        i = j + 1
    return internal_mask, terminal_mask


def build_residue_sample_weights(y):
    internal_mask, terminal_mask = derive_internal_terminal_masks(y)
    sw = np.full(len(y), NEGATIVE_RESIDUE_WEIGHT, dtype=np.float32)
    sw[terminal_mask > 0] = TERMINAL_POSITIVE_WEIGHT
    sw[internal_mask > 0] = INTERNAL_POSITIVE_WEIGHT
    return sw


def summarize_lengths(files, expected_f):
    lengths = []
    for path in files:
        X, y = load_npz_raw(path, expected_f)
        lengths.append(int(X.shape[0]))
    arr = np.asarray(lengths, dtype=np.int32)
    return {
        "min": int(arr.min()),
        "median": int(np.median(arr)),
        "max": int(arr.max()),
    }


def pick_subset(files, frac, rng):
    n = max(1, int(round(len(files) * frac)))
    n = min(n, len(files))
    idx = np.arange(len(files))
    rng.shuffle(idx)
    idx = sorted(idx[:n].tolist())
    return [str(files[i]) for i in idx]


def create_or_load_splits(train_files, holdout_files):
    saved = load_json(SPLITS_PATH)
    if saved is not None:
        return saved

    rng_a = np.random.RandomState(RANDOM_SEED + 11)
    rng_b = np.random.RandomState(RANDOM_SEED + 17)
    rng_c = np.random.RandomState(RANDOM_SEED + 23)

    train_subset = pick_subset(train_files, GA_TRAIN_FRAC, rng_a)

    all_holdout_idx = np.arange(len(holdout_files))
    rng_b.shuffle(all_holdout_idx)
    n_final_val = max(1, int(round(len(holdout_files) * FINAL_VAL_FRAC)))
    n_final_val = min(n_final_val, len(holdout_files) - 1)
    final_val_idx = sorted(all_holdout_idx[:n_final_val].tolist())
    holdout_unused_idx = sorted(all_holdout_idx[n_final_val:].tolist())

    final_val_files = [str(holdout_files[i]) for i in final_val_idx]
    holdout_unused_files = [str(holdout_files[i]) for i in holdout_unused_idx]

    ga_val_n = max(1, int(round(len(holdout_files) * GA_VAL_FRAC)))
    if ga_val_n > len(final_val_files):
        raise RuntimeError("GA val subset is larger than final validation pool. Adjust fractions.")

    final_val_local_idx = np.arange(len(final_val_files))
    rng_c.shuffle(final_val_local_idx)
    ga_val_local_idx = sorted(final_val_local_idx[:ga_val_n].tolist())
    ga_val_files = [final_val_files[i] for i in ga_val_local_idx]

    splits = {
        "ga_train_files": train_subset,
        "ga_val_files": ga_val_files,
        "final_train_files": [str(p) for p in train_files],
        "final_val_files": final_val_files,
        "all_holdout_files": [str(p) for p in holdout_files],
        "holdout_unused_files": holdout_unused_files,
    }
    save_json(SPLITS_PATH, splits)
    return splits


def load_records(file_list, expected_f):
    records = []
    for raw in file_list:
        path = Path(raw)
        X, y = load_npz_raw(path, expected_f)
        records.append({
            "file": path,
            "entry_id": path.stem,
            "X": X,
            "y": (y >= 0.5).astype(np.float32),
            "sw": build_residue_sample_weights(y),
        })
    return records


def streamed_counts_records(records):
    c0 = 0
    c1 = 0
    for rec in records:
        yb = (rec["y"] >= 0.5).astype(np.int32)
        c1 += int(yb.sum())
        c0 += int((1 - yb).sum())
    return np.array([c0, c1], dtype=np.float32)


def compute_cb_weights(counts, betas):
    cb = np.zeros(2, dtype=np.float32)
    for c in range(2):
        beta = float(betas[c])
        n_c = float(counts[c])
        cb[c] = (1.0 - beta) / (1.0 - (beta ** n_c)) if n_c > 0 else 0.0
    if cb.sum() > 0:
        cb *= 2.0 / cb.sum()
    return cb


def weighted_bce(cb_weights):
    cbw = tf.constant(cb_weights, dtype=tf.float32)
    eps = tf.constant(1e-7, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred_clipped = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        w = y_true * cbw[1] + (1.0 - y_true) * cbw[0]
        bce = -(y_true * tf.math.log(y_pred_clipped) + (1.0 - y_true) * tf.math.log(1.0 - y_pred_clipped))
        return w * bce

    return loss


def build_model(input_dim, cb_weights):
    inp = keras.Input(shape=(None, input_dim), name="sequence_input")
    valid_mask = ValidMaskLayer(name="valid_mask")(inp)
    x = layers.SpatialDropout1D(SPATIAL_DROPOUT)(inp)
    x = layers.Bidirectional(
        layers.LSTM(
            LSTM_UNITS,
            return_sequences=True,
            dropout=LSTM_DROPOUT,
            recurrent_dropout=RECURRENT_DROPOUT,
        )
    )(x, mask=valid_mask)
    x = layers.Dense(1, activation="sigmoid")(x)
    out = ResidueOutLayer(name="residue_out")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(LR), loss=weighted_bce(cb_weights))
    return model


def make_dataset(records, selected_idx, shuffle=False, repeat=False):
    selected_idx = np.asarray(selected_idx, dtype=np.int32)
    input_dim = int(len(selected_idx))

    def gen():
        for rec in records:
            yield rec["X"][:, selected_idx], rec["y"], rec["sw"]

    output_signature = (
        tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        buf = min(5000, max(1000, len(records)))
        ds = ds.shuffle(buf, seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.padded_batch(
        BATCH_SIZE,
        padded_shapes=(tf.TensorShape([None, input_dim]), tf.TensorShape([None]), tf.TensorShape([None])),
        padding_values=(tf.constant(PAD_X_VALUE, tf.float32), tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32)),
    )
    if repeat:
        ds = ds.repeat()
    ds = ds.prefetch(PREFETCH_N)
    steps = max(1, math.ceil(len(records) / BATCH_SIZE))
    return ds, steps


def collect_stream_predictions(model, dataset, thr):
    y_true_chunks = []
    y_pred_chunks = []
    y_prob_chunks = []
    for Xb, yb, swb in dataset:
        y_true = (yb.numpy() >= 0.5).astype(np.int32)
        valid = swb.numpy() > 0.0
        probs = model(Xb, training=False).numpy()
        y_pred = (probs > thr).astype(np.int32)
        y_true_chunks.append(y_true[valid])
        y_pred_chunks.append(y_pred[valid])
        y_prob_chunks.append(probs[valid])
    return np.concatenate(y_true_chunks), np.concatenate(y_pred_chunks), np.concatenate(y_prob_chunks)


def compute_label_metrics(y_true_flat, y_pred_flat):
    p, r, f_std, _ = precision_recall_fscore_support(
        y_true_flat,
        y_pred_flat,
        labels=[0, 1],
        zero_division=0,
    )
    if F1_MODE == "mean_pr":
        f = 0.5 * (p + r)
    else:
        denom = p + r
        f = np.where(denom > 0, (2.0 * p * r) / denom, 0.0)
    macro = float(np.mean(f))
    return {
        "macro_f1": float(macro),
        "f1_label0": float(f[0]),
        "f1_label1": float(f[1]),
        "precision_label0": float(p[0]),
        "precision_label1": float(p[1]),
        "recall_label0": float(r[0]),
        "recall_label1": float(r[1]),
        "f1_label0_standard": float(f_std[0]),
        "f1_label1_standard": float(f_std[1]),
    }


def eval_metrics_stream(model, dataset, thr):
    yt, yp, _ = collect_stream_predictions(model, dataset, thr)
    return compute_label_metrics(yt, yp)


def predict_records_in_batches(model, records, selected_idx):
    selected_idx = np.asarray(selected_idx, dtype=np.int32)
    rows = []
    for start in range(0, len(records), BATCH_SIZE):
        chunk = records[start:start + BATCH_SIZE]
        lengths = [int(len(rec["y"])) for rec in chunk]
        max_len = max(lengths)
        X_batch = np.full((len(chunk), max_len, len(selected_idx)), PAD_X_VALUE, dtype=np.float32)
        y_batch = np.zeros((len(chunk), max_len), dtype=np.int8)
        for i, rec in enumerate(chunk):
            seq_len = lengths[i]
            X_batch[i, :seq_len, :] = rec["X"][:, selected_idx]
            y_batch[i, :seq_len] = (rec["y"] >= 0.5).astype(np.int8)
        prob_np = model(X_batch, training=False).numpy()
        for i, rec in enumerate(chunk):
            seq_len = lengths[i]
            rows.append({
                "file": rec["file"],
                "entry_id": rec["entry_id"],
                "length": seq_len,
                "true": y_batch[i, :seq_len].copy(),
                "prob": prob_np[i, :seq_len].copy(),
            })
    return rows


def threshold_prediction_rows(rows, thr):
    out = []
    for row in rows:
        out.append({
            "file": row["file"],
            "entry_id": row["entry_id"],
            "length": row["length"],
            "true": row["true"],
            "prob": row["prob"],
            "pred": (row["prob"] > thr).astype(np.int8),
        })
    return out


def flatten_sequence_key(rows, key, dtype=np.int32):
    if not rows:
        return np.array([], dtype=dtype)
    return np.concatenate([np.asarray(row[key], dtype=dtype) for row in rows], axis=0)


def collect_flat_from_rows(rows, pred_key="pred"):
    y_true_flat = flatten_sequence_key(rows, "true", dtype=np.int32)
    y_pred_flat = flatten_sequence_key(rows, pred_key, dtype=np.int32)
    y_prob_flat = flatten_sequence_key(rows, "prob", dtype=np.float32)
    return y_true_flat, y_pred_flat, y_prob_flat


def find_runs(arr, value):
    arr = np.asarray(arr, dtype=np.int8)
    runs = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] != value:
            i += 1
            continue
        j = i
        while j + 1 < n and arr[j + 1] == value:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted((int(s), int(e)) for s, e in intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + 1:
            merged[-1][1] = max(last_e, e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def merge_runs_by_zero_gap(runs, max_zero_gap):
    if not runs:
        return []
    runs = sorted((int(s), int(e)) for s, e in runs)
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        last_s, last_e = merged[-1]
        zero_gap = s - last_e - 1
        if zero_gap <= int(max_zero_gap):
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def expand_interval_to_min_len(start, end, min_len, n_total):
    s = int(start)
    e = int(end)
    if e - s + 1 >= int(min_len):
        return s, e
    extra = int(min_len) - (e - s + 1)
    add_left = extra // 2
    add_right = extra - add_left
    s2 = max(0, s - add_left)
    e2 = min(n_total - 1, e + add_right)
    current_len = e2 - s2 + 1
    if current_len < int(min_len):
        remaining = int(min_len) - current_len
        if s2 > 0:
            shift_left = min(remaining, s2)
            s2 -= shift_left
            remaining -= shift_left
        if remaining > 0 and e2 < n_total - 1:
            shift_right = min(remaining, (n_total - 1) - e2)
            e2 += shift_right
    return s2, e2


def smooth_label1_runs(labels, min_len=min_residues_1s_smooth):
    arr = np.asarray(labels, dtype=np.int8)
    n = len(arr)
    if n == 0:
        return arr.copy()
    intervals = find_runs(arr, 1)
    if not intervals:
        return arr.copy()
    prev_intervals = None
    max_iter = max(4, len(intervals) * 3)
    for _ in range(max_iter):
        intervals = merge_runs_by_zero_gap(intervals, max_zero_gap=min_len)
        intervals = [expand_interval_to_min_len(s, e, min_len=min_len, n_total=n) for s, e in intervals]
        intervals = merge_intervals(intervals)
        if intervals == prev_intervals:
            break
        prev_intervals = list(intervals)
    out = arr.copy()
    for s, e in intervals:
        out[s:e + 1] = 1
    return out


def add_smoothed_predictions_to_rows(rows):
    for row in rows:
        row["pred_smoothed"] = smooth_label1_runs(row["pred"], min_len=min_residues_1s_smooth)
    return rows


def count_domains_from_labels(labels, min_zero_len=num_residues_0_domain_definition):
    arr = np.asarray(labels, dtype=np.int8)
    count = 0
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] != 0:
            i += 1
            continue
        j = i
        while j + 1 < n and arr[j + 1] == 0:
            j += 1
        if j - i + 1 >= int(min_zero_len):
            count += 1
        i = j + 1
    return count


def collect_domain_count_vectors(rows, pred_key):
    true_counts = []
    pred_counts = []
    for row in rows:
        true_counts.append(count_domains_from_labels(row["true"], min_zero_len=num_residues_0_domain_definition))
        pred_counts.append(count_domains_from_labels(row[pred_key], min_zero_len=num_residues_0_domain_definition))
    return np.asarray(true_counts, dtype=np.int32), np.asarray(pred_counts, dtype=np.int32)


def compute_domain_count_metrics(true_counts, pred_counts):
    true_counts = np.asarray(true_counts, dtype=np.int32)
    pred_counts = np.asarray(pred_counts, dtype=np.int32)
    diff = pred_counts - true_counts
    n = int(len(diff))
    counts = {
        "exact": int(np.sum(diff == 0)),
        "one_too_few": int(np.sum(diff == -1)),
        "one_too_many": int(np.sum(diff == 1)),
        "two_too_few": int(np.sum(diff == -2)),
        "two_too_many": int(np.sum(diff == 2)),
        "three_or_more_too_few": int(np.sum(diff <= -3)),
        "three_or_more_too_many": int(np.sum(diff >= 3)),
    }
    props = {k: float(v / n) for k, v in counts.items()}
    mae = float(np.mean(np.abs(diff)))
    return {"n_sequences": n, "counts": counts, "props": props, "mae": mae}


def plot_domain_count_category_barplot(metrics_dict, title, out_path):
    labels = ["Exact", "1 too few", "1 too many", "2 too few", "2 too many", "3+ too few", "3+ too many"]
    counts = [
        metrics_dict["counts"]["exact"],
        metrics_dict["counts"]["one_too_few"],
        metrics_dict["counts"]["one_too_many"],
        metrics_dict["counts"]["two_too_few"],
        metrics_dict["counts"]["two_too_many"],
        metrics_dict["counts"]["three_or_more_too_few"],
        metrics_dict["counts"]["three_or_more_too_many"],
    ]
    colors = ["#000000", "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.6)
    ymax = max(counts) if counts else 0
    plt.ylim(0, max(1, ymax) * 1.15 + 1)
    for bar, val in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.2, str(val), ha="center", va="bottom", fontsize=9)
    plt.ylabel("Number of Validation Sequences")
    plt.xlabel("Predicted Minus True Domain Count Category")
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def pad_rows(arrays, pad_value, dtype):
    max_len = max(len(arr) for arr in arrays)
    out = np.full((len(arrays), max_len), pad_value, dtype=dtype)
    for i, arr in enumerate(arrays):
        out[i, :len(arr)] = arr
    return out


def collect_first_n_test_triptych(rows, n_sequences):
    selected = rows[:min(n_sequences, len(rows))]
    pred_before_rows = [row["pred"] for row in selected]
    pred_after_rows = [row["pred_smoothed"] for row in selected]
    true_rows = [row["true"] for row in selected]
    return (
        pad_rows(pred_before_rows, pad_value=-1, dtype=np.int8),
        pad_rows(pred_after_rows, pad_value=-1, dtype=np.int8),
        pad_rows(true_rows, pad_value=-1, dtype=np.int8),
    )


def plot_triptych(pred_before_bin, pred_after_bin, true_bin, out_path):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    binary_cmap = ListedColormap(["#d0d0d0", "#000000", "#ff69b4"])
    binary_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], binary_cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, pred_before_bin.shape[0] * 0.12)), sharey=True, constrained_layout=True)
    axes[0].imshow(pred_before_bin, aspect="auto", interpolation="none", cmap=binary_cmap, norm=binary_norm)
    axes[0].set_title("Predicted Labels (Before Smoothing)")
    axes[0].set_xlabel("Residue Index")
    axes[0].set_ylabel("Validation Sequence #")
    axes[1].imshow(pred_after_bin, aspect="auto", interpolation="none", cmap=binary_cmap, norm=binary_norm)
    axes[1].set_title("Predicted Labels (After Smoothing)")
    axes[1].set_xlabel("Residue Index")
    axes[2].imshow(true_bin, aspect="auto", interpolation="none", cmap=binary_cmap, norm=binary_norm)
    axes[2].set_title("True Labels")
    axes[2].set_xlabel("Residue Index")
    fig.suptitle(f"Validation Sequences (first {pred_before_bin.shape[0]}): Pred Before vs Pred After vs True")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def pick_col(fieldnames, candidates):
    lower = {str(c).strip().lower(): c for c in fieldnames}
    for cand in candidates:
        k = cand.lower().strip()
        if k in lower:
            return lower[k]
    for c in fieldnames:
        cl = str(c).strip().lower()
        for cand in candidates:
            if cand.lower().strip() in cl:
                return c
    return None


def normalize_entry_id(x):
    return str(x).strip().upper()


def normalize_sequence(x):
    return "".join(str(x).strip().upper().split())


def load_targ_sequence_lookup(csv_path):
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        entry_col = pick_col(reader.fieldnames, ["Entry ID", "entry_id", "Entry", "ID"])
        seq_col = pick_col(reader.fieldnames, ["Sequence", "sequence"])
        if entry_col is None or seq_col is None:
            return None
        out = {}
        for row in reader:
            entry_raw = str(row[entry_col]).strip()
            seq = normalize_sequence(row[seq_col])
            if not entry_raw or not seq:
                continue
            key = normalize_entry_id(entry_raw)
            if key not in out:
                out[key] = {"Entry ID": entry_raw, "Sequence": seq}
        return out


def labels_to_bitstring(arr):
    return "".join("1" if int(x) == 1 else "0" for x in arr.tolist())


def export_full_val_sequence_predictions_from_rows(pred_rows, lookup, out_csv):
    rows_out = []
    for row in pred_rows:
        pred_labels = labels_to_bitstring(row["pred"])
        key = normalize_entry_id(row["entry_id"])
        if lookup is not None and key in lookup:
            meta = lookup[key]
            seq = meta["Sequence"]
            covered_len = min(len(pred_labels), len(seq))
            rows_out.append({
                "Entry ID": meta["Entry ID"],
                "Sequence": seq[:covered_len],
                "pred_labels": pred_labels[:covered_len],
            })
        else:
            rows_out.append({
                "Entry ID": row["entry_id"],
                "pred_labels": pred_labels,
            })
    rows_out = sorted(rows_out, key=lambda r: str(r["Entry ID"]))
    fieldnames = list(rows_out[0].keys()) if rows_out else ["Entry ID", "pred_labels"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)


def fmt_metric(x):
    return "nan" if not np.isfinite(x) else f"{x:.4f}"


def load_feature_manifest(path, expected_f):
    if not path.exists():
        raise RuntimeError(f"Missing feature manifest: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = ["pruned_feature_idx", "original_feature_idx", "feature_name", "abs_corr", "rank_desc"]
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty feature manifest: {path}")
        missing_cols = [col for col in required if col not in reader.fieldnames]
        if missing_cols:
            raise RuntimeError(f"Feature manifest missing columns {missing_cols}: {path}")
        out = {}
        for row in reader:
            pruned_idx = int(row["pruned_feature_idx"])
            out[pruned_idx] = {
                "pruned_feature_idx": int(row["pruned_feature_idx"]),
                "original_feature_idx": int(row["original_feature_idx"]),
                "feature_name": str(row["feature_name"]),
                "abs_corr": float(row["abs_corr"]),
                "rank_desc": int(float(row["rank_desc"])),
            }
    missing_idx = [i for i in range(expected_f) if i not in out]
    if missing_idx:
        raise RuntimeError(f"Feature manifest missing pruned_feature_idx values for some features: first missing {missing_idx[:10]}")
    return out


def selected_feature_manifest_rows(idxs, feature_manifest_lookup):
    return [feature_manifest_lookup[int(i)] for i in idxs.tolist()]


def feature_names_from_indices(idxs, feature_manifest_lookup):
    return [feature_manifest_lookup[int(i)]["feature_name"] for i in idxs.tolist()]


def write_selected_feature_manifest_csv(out_dir, idxs, feature_manifest_lookup):
    rows = selected_feature_manifest_rows(idxs, feature_manifest_lookup)
    path = out_dir / "selected_feature_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pruned_feature_idx", "original_feature_idx", "feature_name", "abs_corr", "rank_desc"],
        )
        writer.writeheader()
        writer.writerows(rows)


def genome_key(ind):
    return "".join("1" if int(x) else "0" for x in ind)


def key_to_selected_indices(key):
    return np.asarray([i for i, bit in enumerate(key) if bit == "1"], dtype=np.int32)


def selected_indices(ind):
    arr = np.asarray(ind, dtype=np.int8)
    return np.flatnonzero(arr).astype(np.int32)


def seeded_training_context(key, stage):
    digest = hashlib.md5(f"{stage}:{key}".encode("utf-8")).hexdigest()
    seed = RANDOM_SEED + int(digest[:8], 16) % 1000000
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def current_bits_to_flip(generation):
    return max(1, STARTING_BITS_FLIPPED - ((generation - 1) // DECAY_BIT_FLIPS_EVERY) * REDUCE_BIT_FLIPS_BY)


def random_individual(genome_len):
    k = random.randint(MIN_FEATS, min(MAX_FEATS, genome_len))
    idx = random.sample(range(genome_len), k)
    genome = [0] * genome_len
    for i in idx:
        genome[i] = 1
    return creator.IndividualGA(genome)


def repair_individual(ind):
    ones = [i for i, bit in enumerate(ind) if bit == 1]
    zeros = [i for i, bit in enumerate(ind) if bit == 0]
    if len(ones) < MIN_FEATS:
        need = MIN_FEATS - len(ones)
        for i in random.sample(zeros, need):
            ind[i] = 1
    elif len(ones) > MAX_FEATS:
        need = len(ones) - MAX_FEATS
        for i in random.sample(ones, need):
            ind[i] = 0
    return ind


def mutate_individual(ind, bits_to_flip):
    n = min(bits_to_flip, len(ind))
    for i in random.sample(range(len(ind)), n):
        ind[i] = 0 if ind[i] == 1 else 1
    repair_individual(ind)
    return ind


def population_npz_path(generation):
    return POP_DIR / f"generation_{generation:03d}_population.npz"


def population_csv_path(generation):
    return POP_DIR / f"generation_{generation:03d}_population.csv"


def save_population_checkpoint(population, generation):
    genomes = np.asarray(population, dtype=np.int8)
    fitness = np.asarray([ind.fitness.values for ind in population], dtype=np.float32)
    sizes = genomes.sum(axis=1).astype(np.int32)
    np.savez_compressed(population_npz_path(generation), genomes=genomes, fitness=fitness, sizes=sizes, generation=generation)
    with population_csv_path(generation).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["individual_index", "num_features", "label1_rec_after", "label1_prec_after", "macro_f1_after", "selected_feature_indices"])
        for i, ind in enumerate(population, start=1):
            idxs = selected_indices(ind)
            writer.writerow([
                i,
                int(len(idxs)),
                f"{ind.fitness.values[0]:.6f}",
                f"{ind.fitness.values[1]:.6f}",
                f"{ind.fitness.values[2]:.6f}",
                " ".join(str(int(x)) for x in idxs.tolist()),
            ])


def load_population_checkpoint(generation):
    path = population_npz_path(generation)
    if not path.exists():
        raise RuntimeError(f"Missing saved population file: {path}")
    data = np.load(path, allow_pickle=True)
    genomes = data["genomes"]
    fitness = data["fitness"]
    population = []
    for genome, fit in zip(genomes, fitness):
        ind = creator.IndividualGA(genome.astype(np.int8).tolist())
        ind.fitness.values = tuple(float(x) for x in fit.tolist())
        population.append(ind)
    return population


def save_final_population(population):
    genomes = np.asarray(population, dtype=np.int8)
    fitness = np.asarray([ind.fitness.values for ind in population], dtype=np.float32)
    sizes = genomes.sum(axis=1).astype(np.int32)
    np.savez_compressed(POP_DIR / "final_population.npz", genomes=genomes, fitness=fitness, sizes=sizes)
    with (POP_DIR / "final_population.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["individual_index", "num_features", "label1_rec_after", "label1_prec_after", "macro_f1_after", "selected_feature_indices"])
        for i, ind in enumerate(population, start=1):
            idxs = selected_indices(ind)
            writer.writerow([
                i,
                int(len(idxs)),
                f"{ind.fitness.values[0]:.6f}",
                f"{ind.fitness.values[1]:.6f}",
                f"{ind.fitness.values[2]:.6f}",
                " ".join(str(int(x)) for x in idxs.tolist()),
            ])


def load_final_population():
    data = np.load(POP_DIR / "final_population.npz", allow_pickle=True)
    genomes = data["genomes"]
    fitness = data["fitness"]
    population = []
    for genome, fit in zip(genomes, fitness):
        ind = creator.IndividualGA(genome.astype(np.int8).tolist())
        ind.fitness.values = tuple(float(x) for x in fit.tolist())
        population.append(ind)
    return population


def train_model(records_train, records_val, idxs, epochs, patience, seed_key, stage_name):
    configure_tensorflow_runtime()
    seeded_training_context(seed_key, stage_name)
    cb_counts = streamed_counts_records(records_val)
    cb_weights = compute_cb_weights(cb_counts, CB_BETAS) if USE_CB_WEIGHTS else np.array([1.0, 1.0], dtype=np.float32)

    ds_tr_fit, tr_steps = make_dataset(records_train, idxs, shuffle=True, repeat=True)
    ds_va_fit, va_steps = make_dataset(records_val, idxs, shuffle=False, repeat=True)
    ds_va_eval, _ = make_dataset(records_val, idxs, shuffle=False, repeat=False)

    model = build_model(len(idxs), cb_weights)

    train_losses = []
    val_losses = []
    macro_f1s = []
    f1_0s = []
    f1_1s = []
    best_epoch = -1
    best_weights = None
    best_monitor = -np.inf
    no_improve = 0

    for epoch in range(epochs):
        history = model.fit(
            ds_tr_fit,
            validation_data=ds_va_fit,
            steps_per_epoch=tr_steps,
            validation_steps=va_steps,
            epochs=1,
            verbose=0,
        )
        tr_loss = float(history.history["loss"][0])
        va_loss = float(history.history["val_loss"][0])
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        val_metrics = eval_metrics_stream(model, ds_va_eval, PRED_THRESHOLD)
        macro = val_metrics["macro_f1"]
        f10 = val_metrics["f1_label0"]
        f11 = val_metrics["f1_label1"]

        macro_f1s.append(macro)
        f1_0s.append(f10)
        f1_1s.append(f11)

        if macro > best_monitor:
            best_monitor = macro
            best_epoch = epoch + 1
            best_weights = model.get_weights()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_weights is None:
        best_epoch = len(train_losses)
        best_weights = model.get_weights()

    model.set_weights(best_weights)
    final_val_metrics = eval_metrics_stream(model, ds_va_eval, PRED_THRESHOLD)

    history_out = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_macro_f1": macro_f1s,
        "val_f1_0": f1_0s,
        "val_f1_1": f1_1s,
    }
    return model, history_out, best_epoch, final_val_metrics


def build_ga_cache_entry_for_key(key, ga_train_records, ga_val_records):
    idxs = key_to_selected_indices(key)
    model = None
    try:
        model, _, best_epoch, val_metrics_before = train_model(
            ga_train_records,
            ga_val_records,
            idxs,
            epochs=GA_LSTM_EPOCHS,
            patience=GA_LSTM_EARLY_STOPPING,
            seed_key=key,
            stage_name="ga",
        )

        val_prob_rows = predict_records_in_batches(model, ga_val_records, idxs)
        val_rows_thr = threshold_prediction_rows(val_prob_rows, PRED_THRESHOLD)
        val_rows_thr_smoothed = add_smoothed_predictions_to_rows(val_rows_thr)

        val_yt_smoothed = flatten_sequence_key(val_rows_thr_smoothed, "true", dtype=np.int32)
        val_yp_smoothed = flatten_sequence_key(val_rows_thr_smoothed, "pred_smoothed", dtype=np.int32)
        val_metrics_after = compute_label_metrics(val_yt_smoothed, val_yp_smoothed)

        fit = (
            float(val_metrics_after["recall_label1"]),
            float(val_metrics_after["precision_label1"]),
            float(val_metrics_after["macro_f1"]),
        )

        return {
            "fitness": fit,
            "best_epoch": int(best_epoch),
            "num_features": int(len(idxs)),
            "val_before": {
                "macro_f1": float(val_metrics_before["macro_f1"]),
                "label1_prec": float(val_metrics_before["precision_label1"]),
                "label1_rec": float(val_metrics_before["recall_label1"]),
                "label0_f1": float(val_metrics_before["f1_label0"]),
                "label1_f1": float(val_metrics_before["f1_label1"]),
            },
            "val_after": {
                "macro_f1": float(val_metrics_after["macro_f1"]),
                "label1_prec": float(val_metrics_after["precision_label1"]),
                "label1_rec": float(val_metrics_after["recall_label1"]),
                "label0_f1": float(val_metrics_after["f1_label0"]),
                "label1_f1": float(val_metrics_after["f1_label1"]),
            },
        }
    finally:
        if model is not None:
            del model
        tf.keras.backend.clear_session()
        gc.collect()


def init_ga_worker(ga_train_records, ga_val_records):
    global _WORKER_GA_TRAIN_RECORDS, _WORKER_GA_VAL_RECORDS
    warnings.filterwarnings(
        "ignore",
        message=r".*Your input ran out of data; interrupting training.*",
        category=UserWarning,
    )
    configure_tensorflow_runtime()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    _WORKER_GA_TRAIN_RECORDS = ga_train_records
    _WORKER_GA_VAL_RECORDS = ga_val_records


def evaluate_genome_key_worker(key):
    entry = build_ga_cache_entry_for_key(key, _WORKER_GA_TRAIN_RECORDS, _WORKER_GA_VAL_RECORDS)
    return key, entry


def evaluate_genome(ind, ga_train_records, ga_val_records, fitness_cache):
    key = genome_key(ind)
    if key in fitness_cache:
        fit = tuple(fitness_cache[key]["fitness"])
        ind.fitness.values = fit
        return fit
    entry = build_ga_cache_entry_for_key(key, ga_train_records, ga_val_records)
    fitness_cache[key] = entry
    save_pickle(CACHE_PATH, fitness_cache)
    fit = tuple(entry["fitness"])
    ind.fitness.values = fit
    return fit


def evaluate_population(population, ga_train_records, ga_val_records, fitness_cache, progress_prefix=None, executor=None):
    invalid = [ind for ind in population if not ind.fitness.valid]
    if not invalid:
        return

    total = len(invalid)
    processed = 0
    uncached = {}

    for ind in invalid:
        key = genome_key(ind)
        if key in fitness_cache:
            fit = tuple(fitness_cache[key]["fitness"])
            ind.fitness.values = fit
            processed += 1
            if progress_prefix is not None:
                show_progress(progress_prefix, processed, total)
        else:
            uncached.setdefault(key, []).append(ind)

    if not uncached:
        return

    if executor is None:
        for key, inds in uncached.items():
            entry = build_ga_cache_entry_for_key(key, ga_train_records, ga_val_records)
            fitness_cache[key] = entry
            save_pickle(CACHE_PATH, fitness_cache)
            fit = tuple(entry["fitness"])
            for ind in inds:
                ind.fitness.values = fit
            processed += len(inds)
            if progress_prefix is not None:
                show_progress(progress_prefix, processed, total)
        return

    future_to_key = {executor.submit(evaluate_genome_key_worker, key): key for key in uncached}
    for future in as_completed(future_to_key):
        key, entry = future.result()
        fitness_cache[key] = entry
        save_pickle(CACHE_PATH, fitness_cache)
        fit = tuple(entry["fitness"])
        inds = uncached[key]
        for ind in inds:
            ind.fitness.values = fit
        processed += len(inds)
        if progress_prefix is not None:
            show_progress(progress_prefix, processed, total)


def build_initial_population(genome_len):
    return [random_individual(genome_len) for _ in range(POPULATION_SIZE)]


def write_history_rows(rows):
    with HISTORY_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "generation",
            "bits_flipped",
            "avg_macro_f1_after",
            "avg_label1_prec_after",
            "avg_label1_rec_after",
            "best_macro_f1_after",
            "best_label1_prec_after",
            "best_label1_rec_after",
            "avg_size",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_history_rows():
    if not HISTORY_CSV_PATH.exists():
        return []
    with HISTORY_CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_generation(population, generation, bits_flipped, proportion_parents_retained=None, num_parents_retained=None):
    recalls = np.asarray([ind.fitness.values[0] for ind in population], dtype=np.float32)
    precs = np.asarray([ind.fitness.values[1] for ind in population], dtype=np.float32)
    macros = np.asarray([ind.fitness.values[2] for ind in population], dtype=np.float32)
    sizes = np.asarray([sum(ind) for ind in population], dtype=np.float32)

    best_rec_ind = max(population, key=lambda ind: ind.fitness.values[0])
    best_prec_ind = max(population, key=lambda ind: ind.fitness.values[1])
    best_macro_ind = max(population, key=lambda ind: ind.fitness.values[2])

    log_line(
        f"GEN {generation:03d}/{NUM_GENERATIONS:03d} | flips={bits_flipped} | avg_macro={macros.mean():.4f} | avg_prec1={precs.mean():.4f} | avg_rec1={recalls.mean():.4f} | avg_size={sizes.mean():.2f}"
    )
    log_line(
        f"  best_recall -> rec1={best_rec_ind.fitness.values[0]:.4f} prec1={best_rec_ind.fitness.values[1]:.4f} macro={best_rec_ind.fitness.values[2]:.4f} size={sum(best_rec_ind)}"
    )
    log_line(
        f"  best_prec   -> rec1={best_prec_ind.fitness.values[0]:.4f} prec1={best_prec_ind.fitness.values[1]:.4f} macro={best_prec_ind.fitness.values[2]:.4f} size={sum(best_prec_ind)}"
    )
    log_line(
        f"  best_macro  -> rec1={best_macro_ind.fitness.values[0]:.4f} prec1={best_macro_ind.fitness.values[1]:.4f} macro={best_macro_ind.fitness.values[2]:.4f} size={sum(best_macro_ind)}"
    )
    if proportion_parents_retained is not None and num_parents_retained is not None:
        log_line(
            f"  Proportion_parents_retained = {proportion_parents_retained:.4f} ({num_parents_retained}/{POPULATION_SIZE})"
        )

    return {
        "generation": generation,
        "bits_flipped": bits_flipped,
        "avg_macro_f1_after": float(macros.mean()),
        "avg_label1_prec_after": float(precs.mean()),
        "avg_label1_rec_after": float(recalls.mean()),
        "best_macro_f1_after": float(macros.max()),
        "best_label1_prec_after": float(precs.max()),
        "best_label1_rec_after": float(recalls.max()),
        "avg_size": float(sizes.mean()),
    }


def update_history(row):
    rows = load_history_rows()
    rows = [r for r in rows if int(r["generation"]) != int(row["generation"])]
    rows.append({k: row[k] for k in row})
    rows = sorted(rows, key=lambda r: int(r["generation"]))
    write_history_rows(rows)
    return rows


def plot_history(history_rows):
    if not history_rows:
        return

    gens = [int(r["generation"]) for r in history_rows]
    avg_macro = [float(r["avg_macro_f1_after"]) for r in history_rows]
    avg_prec = [float(r["avg_label1_prec_after"]) for r in history_rows]
    avg_rec = [float(r["avg_label1_rec_after"]) for r in history_rows]
    avg_size = [float(r["avg_size"]) for r in history_rows]
    best_macro = [float(r["best_macro_f1_after"]) for r in history_rows]
    best_prec = [float(r["best_label1_prec_after"]) for r in history_rows]
    best_rec = [float(r["best_label1_rec_after"]) for r in history_rows]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(gens, avg_macro, label="Avg Macro F1", color="#000000")
    ax1.plot(gens, avg_prec, label="Avg Label 1 Precision", color="#f58518")
    ax1.plot(gens, avg_rec, label="Avg Label 1 Recall", color="#4c78a8")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Metric")
    ax2 = ax1.twinx()
    ax2.plot(gens, avg_size, label="Avg Size", color="#ff69b4", linestyle="--")
    ax2.set_ylabel("Average Feature Count")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.tight_layout()
    plt.savefig(AVG_PLOT_PATH, dpi=300)
    plt.close(fig)

    plt.figure(figsize=(10, 5.5))
    plt.plot(gens, best_macro, label="Best Macro F1", color="#000000")
    plt.plot(gens, best_prec, label="Best Label 1 Precision", color="#f58518")
    plt.plot(gens, best_rec, label="Best Label 1 Recall", color="#4c78a8")
    plt.xlabel("Generation")
    plt.ylabel("Metric")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(BEST_PLOT_PATH, dpi=300)
    plt.close()


def bottom_replace(population, genome_len, ga_train_records, ga_val_records, fitness_cache, generation, executor=None):
    n_replace = max(1, int(round(len(population) * BOTTOM_FRAC_REPLACE)))
    ranked = sorted(range(len(population)), key=lambda i: population[i].fitness.values[2])
    replace_idx = ranked[:n_replace]
    for idx in replace_idx:
        population[idx] = random_individual(genome_len)
    evaluate_population(
        [population[i] for i in replace_idx],
        ga_train_records,
        ga_val_records,
        fitness_cache,
        progress_prefix=f"Replacing bottom at gen {generation}/{NUM_GENERATIONS}",
        executor=executor,
    )
    return n_replace


def make_reference_points():
    return tools.uniform_reference_points(nobj=3, p=REFERENCE_POINTS)


def evolve_one_generation(population, generation, genome_len, ga_train_records, ga_val_records, fitness_cache, ref_points, executor=None):
    bits_to_flip = current_bits_to_flip(generation)
    parents = [creator.IndividualGA(ind[:]) for ind in tools.selRandom(population, len(population))]
    offspring = []

    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]

        child1 = creator.IndividualGA(parent1[:])
        child2 = creator.IndividualGA(parent2[:])
        tools.cxTwoPoint(child1, child2)
        mutate_individual(child1, bits_to_flip)
        mutate_individual(child2, bits_to_flip)
        if child1.fitness.valid:
            del child1.fitness.values
        if child2.fitness.valid:
            del child2.fitness.values
        offspring.append(child1)
        if len(offspring) < len(parents):
            offspring.append(child2)

    evaluate_population(
        offspring,
        ga_train_records,
        ga_val_records,
        fitness_cache,
        progress_prefix=f"Evaluating offspring for gen {generation}/{NUM_GENERATIONS}",
        executor=executor,
    )

    parent_ids = {id(ind) for ind in population}
    combined = population + offspring
    selected_population = tools.selNSGA3(combined, POPULATION_SIZE, ref_points)
    num_parents_retained = sum(1 for ind in selected_population if id(ind) in parent_ids)
    proportion_parents_retained = num_parents_retained / POPULATION_SIZE
    population = selected_population

    replaced = 0
    if generation % NUM_GEN_REPLACE == 0 and generation < NUM_GENERATIONS:
        replaced = bottom_replace(population, genome_len, ga_train_records, ga_val_records, fitness_cache, generation, executor=executor)

    row = summarize_generation(population, generation, bits_to_flip, proportion_parents_retained, num_parents_retained)
    if replaced > 0:
        log_line(f"  replaced_bottom_macro={replaced}")

    save_population_checkpoint(population, generation)
    history_rows = update_history(row)
    plot_history(history_rows)
    save_rng_state()

    if SYNC_EVERY_GENERATION:
        maybe_run_sync_hook("generation_complete", {"GA_GENERATION": generation})

    return population


def write_state(state):
    save_json(STATE_PATH, state)


def load_state():
    return load_json(STATE_PATH)


def initialize_run(genome_len, ga_train_records, ga_val_records, fitness_cache, executor=None):
    population = build_initial_population(genome_len)
    evaluate_population(
        population,
        ga_train_records,
        ga_val_records,
        fitness_cache,
        progress_prefix="Initializing population",
        executor=executor,
    )
    row = summarize_generation(population, 0, current_bits_to_flip(1))
    save_population_checkpoint(population, 0)
    history_rows = update_history(row)
    plot_history(history_rows)
    save_rng_state()
    write_state({
        "phase": "ga",
        "last_completed_generation": 0,
        "next_generation": 1,
    })
    if SYNC_EVERY_GENERATION:
        maybe_run_sync_hook("generation_complete", {"GA_GENERATION": 0})
    return population


def run_ga(genome_len, ga_train_records, ga_val_records):
    fitness_cache = load_pickle(CACHE_PATH, default={})
    state = load_state()
    ref_points = make_reference_points()
    executor = None
    eff_workers = effective_max_workers()

    try:
        if MAX_WORKERS > 1 and eff_workers == 1 and gpu_is_available() and FORCE_SERIAL_ON_GPU:
            log_line(
                f"GPU detected; forcing serial GA evaluation. requested_MAX_WORKERS={MAX_WORKERS} effective_MAX_WORKERS={eff_workers}"
            )

        if eff_workers > 1:
            log_line(f"Using CPU parallel evaluation with MAX_WORKERS={eff_workers}")
            executor = ProcessPoolExecutor(
                max_workers=eff_workers,
                mp_context=get_context("spawn"),
                initializer=init_ga_worker,
                initargs=(ga_train_records, ga_val_records),
            )
        else:
            log_line("Using single-process evaluation")

        if state is None:
            load_rng_state()
            population = initialize_run(genome_len, ga_train_records, ga_val_records, fitness_cache, executor=executor)
            state = load_state()
        else:
            population = load_population_checkpoint(state["last_completed_generation"])
            load_rng_state()
            log_line(f"Resuming GA from generation {state['next_generation']}/{NUM_GENERATIONS}")

        for generation in range(state["next_generation"], NUM_GENERATIONS + 1):
            log_line(f"Generation {generation}/{NUM_GENERATIONS}")
            population = evolve_one_generation(
                population,
                generation,
                genome_len,
                ga_train_records,
                ga_val_records,
                fitness_cache,
                ref_points,
                executor=executor,
            )
            if generation == NUM_GENERATIONS:
                save_final_population(population)
                write_state({
                    "phase": "final_retrain",
                    "last_completed_generation": generation,
                    "next_generation": generation + 1,
                })
                save_json(FINAL_RETRAIN_STATE_PATH, {"completed_indices": []})
                log_line("GA complete. Final population saved.")
                maybe_run_sync_hook("ga_complete", {"GA_GENERATION": generation})
            else:
                write_state({
                    "phase": "ga",
                    "last_completed_generation": generation,
                    "next_generation": generation + 1,
                })
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def write_history_csv(path, history_dict):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_macro_f1", "val_f1_0", "val_f1_1"])
        writer.writeheader()
        for i in range(len(history_dict["train_loss"])):
            writer.writerow({
                "epoch": i + 1,
                "train_loss": f"{history_dict['train_loss'][i]:.6f}",
                "val_loss": f"{history_dict['val_loss'][i]:.6f}",
                "val_macro_f1": f"{history_dict['val_macro_f1'][i]:.6f}",
                "val_f1_0": f"{history_dict['val_f1_0'][i]:.6f}",
                "val_f1_1": f"{history_dict['val_f1_1'][i]:.6f}",
            })


def plot_training_curves(history_dict, out_dir):
    plt.figure()
    plt.plot(history_dict["train_loss"], label="Train Loss")
    plt.plot(history_dict["val_loss"], label="Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history_dict["val_macro_f1"], label="VAL Macro F1")
    plt.plot(history_dict["val_f1_0"], label="VAL Label 0 F1")
    plt.plot(history_dict["val_f1_1"], label="VAL Label 1 F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "f1_curves.png", dpi=300)
    plt.close()


def write_selected_features_json(out_dir, idxs, feature_manifest_lookup):
    rows = selected_feature_manifest_rows(idxs, feature_manifest_lookup)
    payload = {
        "num_features": int(len(idxs)),
        "selected_feature_indices": [int(x) for x in idxs.tolist()],
        "selected_feature_names": [row["feature_name"] for row in rows],
        "selected_feature_manifest_rows": rows,
    }
    save_json(out_dir / "selected_features.json", payload)


def format_threshold_label(thr):
    return f"{thr:.3f}".rstrip("0").rstrip(".")


def make_threshold_subdir(individual_dir, thr):
    return individual_dir / format_threshold_label(thr)


def individuals_threshold_csv_path(thr):
    return INDIV_DIR / f"pred{format_threshold_label(thr)}_individuals_results.csv"


THRESHOLD_METRICS_FIELDNAMES = [
    "individual_id",
    "num_features",
    "best_epoch",
    "pred_threshold",
    "label0_f1_before",
    "label1_f1_before",
    "label1_prec_before",
    "label1_rec_before",
    "macro_f1_before",
    "domain_count_mae_before",
    "label0_f1_after",
    "label1_f1_after",
    "label1_prec_after",
    "label1_rec_after",
    "macro_f1_after",
    "domain_count_mae_after",
]


def fmt_csv_metric(x):
    return "" if not np.isfinite(x) else f"{float(x):.6f}"


def compute_threshold_metrics(rows_thr):
    yt, yp, _ = collect_flat_from_rows(rows_thr, pred_key="pred")
    metrics_before = compute_label_metrics(yt, yp)

    rows_thr_smoothed = add_smoothed_predictions_to_rows(rows_thr)
    yt_smoothed = flatten_sequence_key(rows_thr_smoothed, "true", dtype=np.int32)
    yp_smoothed = flatten_sequence_key(rows_thr_smoothed, "pred_smoothed", dtype=np.int32)
    metrics_after = compute_label_metrics(yt_smoothed, yp_smoothed)

    return rows_thr_smoothed, metrics_before, metrics_after


def build_threshold_metrics_row(individual_id, num_features, best_epoch, thr, metrics_before, metrics_after, domain_before, domain_after):
    return {
        "individual_id": individual_id,
        "num_features": str(int(num_features)),
        "best_epoch": str(int(best_epoch)),
        "pred_threshold": format_threshold_label(thr),
        "label0_f1_before": fmt_csv_metric(metrics_before["f1_label0"]),
        "label1_f1_before": fmt_csv_metric(metrics_before["f1_label1"]),
        "label1_prec_before": fmt_csv_metric(metrics_before["precision_label1"]),
        "label1_rec_before": fmt_csv_metric(metrics_before["recall_label1"]),
        "macro_f1_before": fmt_csv_metric(metrics_before["macro_f1"]),
        "domain_count_mae_before": fmt_csv_metric(domain_before["mae"]),
        "label0_f1_after": fmt_csv_metric(metrics_after["f1_label0"]),
        "label1_f1_after": fmt_csv_metric(metrics_after["f1_label1"]),
        "label1_prec_after": fmt_csv_metric(metrics_after["precision_label1"]),
        "label1_rec_after": fmt_csv_metric(metrics_after["recall_label1"]),
        "macro_f1_after": fmt_csv_metric(metrics_after["macro_f1"]),
        "domain_count_mae_after": fmt_csv_metric(domain_after["mae"]),
    }


def write_threshold_metrics_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=THRESHOLD_METRICS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def load_threshold_metrics_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rebuild_individuals_threshold_csvs():
    rows_by_threshold = {format_threshold_label(thr): [] for thr in FINAL_EVAL_THRESHOLDS}

    for individual_dir in sorted(p for p in INDIV_DIR.glob("individual_*") if p.is_dir()):
        summary_path = individual_dir / INDIVIDUAL_THRESHOLD_SUMMARY_CSV
        for row in load_threshold_metrics_csv(summary_path):
            thr_label = row.get("pred_threshold", "")
            if thr_label in rows_by_threshold:
                rows_by_threshold[thr_label].append(row)

    for thr in FINAL_EVAL_THRESHOLDS:
        thr_label = format_threshold_label(thr)
        rows = sorted(rows_by_threshold[thr_label], key=lambda r: r["individual_id"])
        write_threshold_metrics_csv(individuals_threshold_csv_path(thr), rows)


def write_final_summary(out_dir, num_features, best_epoch, pred_threshold, val_metrics_before, val_metrics_after, domain_before, domain_after):
    text = (
        f"NUM FEATURES: {num_features}\n"
        f"BEST EPOCH: {best_epoch}\n"
        f"EARLY_STOP_MODE: val_macro_f1\n"
        f"F1_MODE: {F1_MODE}\n"
        f"PRED_THRESHOLD: {pred_threshold:g}\n"
        f"VAL BEFORE SMOOTHING Macro F1: {val_metrics_before['macro_f1']:.4f}\n"
        f"VAL BEFORE SMOOTHING F1 label0: {val_metrics_before['f1_label0']:.4f}\n"
        f"VAL BEFORE SMOOTHING F1 label1: {val_metrics_before['f1_label1']:.4f}\n"
        f"VAL BEFORE SMOOTHING Precision label1: {val_metrics_before['precision_label1']:.4f}\n"
        f"VAL BEFORE SMOOTHING Recall label1: {val_metrics_before['recall_label1']:.4f}\n"
        f"VAL AFTER SMOOTHING Macro F1: {val_metrics_after['macro_f1']:.4f}\n"
        f"VAL AFTER SMOOTHING F1 label0: {val_metrics_after['f1_label0']:.4f}\n"
        f"VAL AFTER SMOOTHING F1 label1: {val_metrics_after['f1_label1']:.4f}\n"
        f"VAL AFTER SMOOTHING Precision label1: {val_metrics_after['precision_label1']:.4f}\n"
        f"VAL AFTER SMOOTHING Recall label1: {val_metrics_after['recall_label1']:.4f}\n"
        f"VAL Domain-count exact proportion before smoothing: {fmt_metric(domain_before['props']['exact'])}\n"
        f"VAL Domain-count one-too-few proportion before smoothing: {fmt_metric(domain_before['props']['one_too_few'])}\n"
        f"VAL Domain-count one-too-many proportion before smoothing: {fmt_metric(domain_before['props']['one_too_many'])}\n"
        f"VAL Domain-count two-too-few proportion before smoothing: {fmt_metric(domain_before['props']['two_too_few'])}\n"
        f"VAL Domain-count two-too-many proportion before smoothing: {fmt_metric(domain_before['props']['two_too_many'])}\n"
        f"VAL Domain-count three-or-more-too-few proportion before smoothing: {fmt_metric(domain_before['props']['three_or_more_too_few'])}\n"
        f"VAL Domain-count three-or-more-too-many proportion before smoothing: {fmt_metric(domain_before['props']['three_or_more_too_many'])}\n"
        f"VAL Domain-count MAE before smoothing: {fmt_metric(domain_before['mae'])}\n"
        f"VAL Domain-count exact proportion after smoothing: {fmt_metric(domain_after['props']['exact'])}\n"
        f"VAL Domain-count one-too-few proportion after smoothing: {fmt_metric(domain_after['props']['one_too_few'])}\n"
        f"VAL Domain-count one-too-many proportion after smoothing: {fmt_metric(domain_after['props']['one_too_many'])}\n"
        f"VAL Domain-count two-too-few proportion after smoothing: {fmt_metric(domain_after['props']['two_too_few'])}\n"
        f"VAL Domain-count two-too-many proportion after smoothing: {fmt_metric(domain_after['props']['two_too_many'])}\n"
        f"VAL Domain-count three-or-more-too-few proportion after smoothing: {fmt_metric(domain_after['props']['three_or_more_too_few'])}\n"
        f"VAL Domain-count three-or-more-too-many proportion after smoothing: {fmt_metric(domain_after['props']['three_or_more_too_many'])}\n"
        f"VAL Domain-count MAE after smoothing: {fmt_metric(domain_after['mae'])}\n"
        f"DOMAIN COUNT RULE: number of label-0 stretches with length >= {num_residues_0_domain_definition}\n"
        f"BOUNDARY SMOOTHING RULE: predicted label-1 runs are merged if the zero-gap between them <= {min_residues_1s_smooth}, and any resulting label-1 run shorter than {min_residues_1s_smooth} is expanded outward to length {min_residues_1s_smooth}\n"
    )
    (out_dir / "final_summary.txt").write_text(text, encoding="utf-8")


def retrain_one_final_individual(individual_index, ind, final_train_records, final_val_records, all_holdout_records, lookup, feature_manifest_lookup):
    out_dir = INDIV_DIR / f"individual_{individual_index:03d}"
    marker = out_dir / "completed.marker"
    if marker.exists():
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = selected_indices(ind)
    key = genome_key(ind)
    individual_id = f"individual_{individual_index:03d}"
    log_line(f"Final retrain {individual_index}/{POPULATION_SIZE} | features={len(idxs)}")

    model = None
    try:
        model, history_dict, best_epoch, _ = train_model(
            final_train_records,
            final_val_records,
            idxs,
            epochs=FINAL_LSTM_EPOCHS,
            patience=FINAL_LSTM_EARLY_STOPPING,
            seed_key=key,
            stage_name="final",
        )

        val_prob_rows = predict_records_in_batches(model, final_val_records, idxs)
        all_holdout_prob_rows = predict_records_in_batches(model, all_holdout_records, idxs)

        threshold_rows = []
        for thr in FINAL_EVAL_THRESHOLDS:
            thr_label = format_threshold_label(thr)
            threshold_dir = make_threshold_subdir(out_dir, thr)
            threshold_dir.mkdir(parents=True, exist_ok=True)

            val_rows_thr = threshold_prediction_rows(val_prob_rows, thr)
            val_rows_thr_smoothed, val_metrics_before, val_metrics_after = compute_threshold_metrics(val_rows_thr)

            domain_true_before, domain_pred_before = collect_domain_count_vectors(val_rows_thr, pred_key="pred")
            domain_true_after, domain_pred_after = collect_domain_count_vectors(val_rows_thr_smoothed, pred_key="pred_smoothed")
            domain_before = compute_domain_count_metrics(domain_true_before, domain_pred_before)
            domain_after = compute_domain_count_metrics(domain_true_after, domain_pred_after)

            plot_domain_count_category_barplot(
                domain_before,
                f"Validation Domain-Count Error Categories (Before Smoothing, thr={thr_label})",
                threshold_dir / DOMAIN_COUNT_BARPLOT_BEFORE_FILE,
            )
            plot_domain_count_category_barplot(
                domain_after,
                f"Validation Domain-Count Error Categories (After Smoothing, thr={thr_label})",
                threshold_dir / DOMAIN_COUNT_BARPLOT_AFTER_FILE,
            )

            pred_before_n, pred_after_n, true_n = collect_first_n_test_triptych(val_rows_thr_smoothed, TEST_HEATMAP_N)
            plot_triptych(pred_before_n, pred_after_n, true_n, threshold_dir / TEST_HEATMAP_FILE)

            all_holdout_rows_thr = threshold_prediction_rows(all_holdout_prob_rows, thr)
            export_full_val_sequence_predictions_from_rows(all_holdout_rows_thr, lookup, threshold_dir / FULL_VAL_SEQUENCE_PRED_CSV)

            write_final_summary(
                threshold_dir,
                len(idxs),
                best_epoch,
                thr,
                val_metrics_before,
                val_metrics_after,
                domain_before,
                domain_after,
            )

            threshold_rows.append(
                build_threshold_metrics_row(
                    individual_id,
                    len(idxs),
                    best_epoch,
                    thr,
                    val_metrics_before,
                    val_metrics_after,
                    domain_before,
                    domain_after,
                )
            )
            log_line(
                f"  thr={thr_label} | val_macro_after={val_metrics_after['macro_f1']:.4f} | val_prec1_after={val_metrics_after['precision_label1']:.4f} | val_rec1_after={val_metrics_after['recall_label1']:.4f}"
            )

        write_threshold_metrics_csv(out_dir / INDIVIDUAL_THRESHOLD_SUMMARY_CSV, threshold_rows)
        write_history_csv(out_dir / "history.csv", history_dict)
        plot_training_curves(history_dict, out_dir)
        write_selected_features_json(out_dir, idxs, feature_manifest_lookup)
        write_selected_feature_manifest_csv(out_dir, idxs, feature_manifest_lookup)
        model.save(out_dir / "model_best.keras")
        marker.touch()

        if SYNC_EVERY_INDIVIDUAL:
            maybe_run_sync_hook("individual_complete", {"GA_INDIVIDUAL_ID": individual_id})

        return True
    finally:
        if model is not None:
            del model
        tf.keras.backend.clear_session()
        gc.collect()


def run_final_retrain(expected_f, splits, feature_manifest_lookup):
    population = load_final_population()
    lookup = load_targ_sequence_lookup(TARG_DEDUPED_CSV)
    final_train_records = load_records(splits["final_train_files"], expected_f)
    final_val_records = load_records(splits["final_val_files"], expected_f)
    all_holdout_records = load_records(splits["all_holdout_files"], expected_f)
    retrain_state = load_json(FINAL_RETRAIN_STATE_PATH, default={"completed_indices": []})
    completed = set(int(x) for x in retrain_state.get("completed_indices", []))

    rebuild_individuals_threshold_csvs()

    for i, ind in enumerate(population, start=1):
        if i in completed and (INDIV_DIR / f"individual_{i:03d}" / "completed.marker").exists():
            continue
        ok = retrain_one_final_individual(i, ind, final_train_records, final_val_records, all_holdout_records, lookup, feature_manifest_lookup)
        if ok:
            completed.add(i)
            save_json(FINAL_RETRAIN_STATE_PATH, {"completed_indices": sorted(completed)})
            rebuild_individuals_threshold_csvs()
            write_state({
                "phase": "final_retrain",
                "last_completed_generation": NUM_GENERATIONS,
                "next_generation": NUM_GENERATIONS + 1,
                "last_completed_individual": i,
            })

    rebuild_individuals_threshold_csvs()
    write_state({
        "phase": "done",
        "last_completed_generation": NUM_GENERATIONS,
        "next_generation": NUM_GENERATIONS + 1,
        "last_completed_individual": POPULATION_SIZE,
    })
    log_line("Final retrain complete.")
    maybe_run_sync_hook("run_complete")


def main():
    configure_tensorflow_runtime()
    ensure_dirs()
    verify_or_write_config()
    log_runtime_info()

    train_files = list_npz_files(TRAIN_CACHE_DIR)
    holdout_files = list_npz_files(TEST_AND_VAL_CACHE_DIR)

    expected_f = inspect_feature_dim_from_first(train_files)
    holdout_f = inspect_feature_dim_from_first(holdout_files)

    if expected_f != holdout_f:
        raise RuntimeError(f"Feature dim mismatch: train={expected_f}, test_and_val={holdout_f}")
    if MAX_FEATS > expected_f:
        raise RuntimeError(f"MAX_FEATS={MAX_FEATS} exceeds available feature count F={expected_f}")
    if MIN_FEATS < 1 or MIN_FEATS > MAX_FEATS:
        raise RuntimeError("Invalid MIN_FEATS/MAX_FEATS settings")

    train_len_stats = summarize_lengths(train_files, expected_f)
    holdout_len_stats = summarize_lengths(holdout_files, expected_f)
    log_line(f"train files: {len(train_files)} | test_and_val files: {len(holdout_files)}")
    log_line(f"features per position: {expected_f}")
    log_line(
        f"train lengths min/median/max = {train_len_stats['min']}/{train_len_stats['median']}/{train_len_stats['max']}"
    )
    log_line(
        f"test_and_val lengths min/median/max = {holdout_len_stats['min']}/{holdout_len_stats['median']}/{holdout_len_stats['max']}"
    )

    feature_manifest_lookup = load_feature_manifest(FEATURE_MANIFEST_CSV, expected_f)
    log_line(f"feature manifest rows: {len(feature_manifest_lookup)}")

    splits = create_or_load_splits(train_files, holdout_files)
    state = load_state()

    if state is not None and state.get("phase") == "done":
        log_line("GA and final retrain are already complete.")
        return

    if state is None or state.get("phase") == "ga":
        ga_train_records = load_records(splits["ga_train_files"], expected_f)
        ga_val_records = load_records(splits["ga_val_files"], expected_f)
        run_ga(expected_f, ga_train_records, ga_val_records)
        state = load_state()

    if state is not None and state.get("phase") == "final_retrain":
        run_final_retrain(expected_f, splits, feature_manifest_lookup)


if __name__ == "__main__":
    main()
