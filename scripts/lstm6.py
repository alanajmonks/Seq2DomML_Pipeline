from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"RUNNING FILE: {Path(__file__).resolve()}", flush=True)
print("SCRIPT VERSION: manual-batch-v2-lazy-postimports", flush=True)

TRAIN_CACHE_DIR = Path("data/pruned_featuresets/1.0/train")
HOLDOUT_CACHE_DIR = Path("data/pruned_featuresets/1.0/test_and_val")

PAD_X_VALUE = -999.0
BATCH_SIZE = 50
EPOCHS = 500
LR = 1e-4
LSTM_UNITS = 100
SPATIAL_DROPOUT = 0.1
LSTM_DROPOUT = 0.1
RECURRENT_DROPOUT = 0.0
USE_FOCAL = False
FOCAL_GAMMA = 1.5
USE_CB_WEIGHTS = True
CB_BETAS = np.array([0.99999, 0.99999], dtype=np.float32)

EARLY_STOP_PATIENCE = 10
EARLY_STOP_VAL_MACRO = True
F1_MODE = "mean_pr"

VAL_FRAC = 0.35
PRED_THRESHOLD = 0.55
THRESHOLD_SWEEP = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

TEST_HEATMAP_N = 20
TEST_HEATMAP_FILE = "heatmap_test25_pred_before_after_true_triptych.png"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

OUT_DIR = Path("results/corprune_trials/1.0")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARG_DEDUPED_CSV = Path("data/preprocessing/targ_deduped.csv")
FULL_VAL_SEQUENCE_PRED_CSV = "full_val_sequence_predictions.csv"

NEGATIVE_RESIDUE_WEIGHT = 0.5
TERMINAL_POSITIVE_WEIGHT = 0.25
INTERNAL_POSITIVE_WEIGHT = 1.0

min_residues_1s_smooth = 9
num_residues_0_domain_definition = 10

DOMAIN_COUNT_BARPLOT_BEFORE_FILE = "barplot_test_domain_count_categories_before_smoothing.png"
DOMAIN_COUNT_BARPLOT_AFTER_FILE = "barplot_test_domain_count_categories_after_smoothing.png"


def list_npz_files(path: Path):
    files = sorted(path.glob("*.npz"))
    if not files:
        raise RuntimeError(f"no npz files in {path}")
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


def summarize_lengths(files):
    lengths = []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            X = d["X"]
            y = d["y"]
        if X.ndim != 2 or y.ndim != 1:
            raise RuntimeError(f"{f.name}: bad shapes X={X.shape}, y={y.shape}")
        if X.shape[0] != y.shape[0]:
            raise RuntimeError(f"{f.name}: length mismatch X[0]={X.shape[0]} vs y[0]={y.shape[0]}")
        lengths.append(int(X.shape[0]))
    arr = np.asarray(lengths, dtype=np.int32)
    return {
        "min": int(arr.min()),
        "median": int(np.median(arr)),
        "max": int(arr.max()),
    }


def load_npz_raw(f: Path):
    with np.load(f, allow_pickle=True) as d:
        X = d["X"].astype(np.float32, copy=False)
        y = d["y"].astype(np.float32, copy=False)
    if X.ndim != 2:
        raise RuntimeError(f"{f.name}: expected X.ndim==2 (got {X.ndim}), shape={X.shape}")
    if y.ndim != 1:
        raise RuntimeError(f"{f.name}: expected y.ndim==1 (got {y.ndim}), shape={y.shape}")
    if X.shape[0] != y.shape[0]:
        raise RuntimeError(f"{f.name}: length mismatch X[0]={X.shape[0]} vs y[0]={y.shape[0]}")
    if X.shape[0] <= 0:
        raise RuntimeError(f"{f.name}: empty sequence is not allowed")
    if X.shape[1] != F:
        raise RuntimeError(f"{f.name}: expected F={F}, got X.shape[1]={X.shape[1]}")
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

        if i == 0 or j == (n - 1):
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


train_resolved = TRAIN_CACHE_DIR.resolve()
holdout_resolved = HOLDOUT_CACHE_DIR.resolve()
if train_resolved == holdout_resolved:
    raise RuntimeError("train and holdout directories resolve to the same path")

print("listing train sequences")
tr_files = list_npz_files(TRAIN_CACHE_DIR)

print("listing test_and_val sequences")
all_files = list_npz_files(HOLDOUT_CACHE_DIR)

F = inspect_feature_dim_from_first(tr_files)
F2 = inspect_feature_dim_from_first(all_files)
if F2 != F:
    raise RuntimeError(f"feature dim mismatch: train F={F} vs test_and_val F={F2}")

Ntr = len(tr_files)
Nall = len(all_files)
train_len_stats = summarize_lengths(tr_files)
all_len_stats = summarize_lengths(all_files)

print(f"train sequences: {Ntr} | test_and_val sequences: {Nall}")
print(f"features per position: {F}")
print(f"train lengths min/median/max = {train_len_stats['min']}/{train_len_stats['median']}/{train_len_stats['max']}")
print(f"test_and_val lengths min/median/max = {all_len_stats['min']}/{all_len_stats['median']}/{all_len_stats['max']}")

idx = np.arange(Nall)
rng = np.random.RandomState(RANDOM_SEED)
rng.shuffle(idx)

n_val = max(1, int(round(Nall * VAL_FRAC)))
n_val = min(n_val, Nall - 1)

val_idx = idx[:n_val]
test_idx = idx[n_val:]

val_files = [all_files[i] for i in val_idx]
test_files = [all_files[i] for i in test_idx]

print(f"val sequences: {len(val_files)} | test sequences: {len(test_files)}")


def make_padded_batch(files):
    lengths = []
    X_list = []
    y_list = []
    sw_list = []

    for f in files:
        X, y = load_npz_raw(f)
        sw_residue = build_residue_sample_weights(y)
        lengths.append(int(X.shape[0]))
        X_list.append(X)
        y_list.append(y)
        sw_list.append(sw_residue)

    max_len = max(lengths)

    X_batch = np.full((len(files), max_len, F), PAD_X_VALUE, dtype=np.float32)
    y_batch = np.zeros((len(files), max_len), dtype=np.float32)
    sw_batch = np.zeros((len(files), max_len), dtype=np.float32)

    for i, (X, y, sw, seq_len) in enumerate(zip(X_list, y_list, sw_list, lengths)):
        X_batch[i, :seq_len, :] = X
        y_batch[i, :seq_len] = y
        sw_batch[i, :seq_len] = sw

    return X_batch, y_batch, sw_batch


def iter_batch_file_lists(files, batch_size, shuffle=False, seed=None):
    order = np.arange(len(files), dtype=np.int32)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(order)

    for start in range(0, len(order), batch_size):
        batch_idx = order[start:start + batch_size]
        yield [files[i] for i in batch_idx]


def streamed_counts(files):
    c0 = 0
    c1 = 0
    for f in files:
        _, y = load_npz_raw(f)
        yb = (y >= 0.5).astype(np.int32)
        c1 += int(yb.sum())
        c0 += int((1 - yb).sum())
    return np.array([c0, c1], dtype=np.float32)


counts = streamed_counts(tr_files)


def compute_cb_weights(counts, betas):
    cb = np.zeros(2, dtype=np.float32)
    for c in range(2):
        beta = float(betas[c])
        n_c = float(counts[c])
        cb[c] = (1.0 - beta) / (1.0 - (beta ** n_c)) if n_c > 0 else 0.0
    if cb.sum() > 0:
        cb *= 2.0 / cb.sum()
    return cb


cb_weights = compute_cb_weights(counts, CB_BETAS)
if not USE_CB_WEIGHTS:
    cb_weights = np.array([1.0, 1.0], dtype=np.float32)

print(f"train residue counts: {counts.tolist()}")
print(f"class weights: {cb_weights.tolist()}")


def weighted_bce(cb_weights):
    cbw = tf.constant(cb_weights, dtype=tf.float32)
    eps = tf.constant(1e-7, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        w = y_true * cbw[1] + (1.0 - y_true) * cbw[0]
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return w * bce

    return loss


def weighted_focal(cb_weights, gamma):
    cbw = tf.constant(cb_weights, dtype=tf.float32)
    eps = tf.constant(1e-7, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        w = y_true * cbw[1] + (1.0 - y_true) * cbw[0]
        return -w * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)

    return loss


loss_fn = weighted_focal(cb_weights, FOCAL_GAMMA) if USE_FOCAL else weighted_bce(cb_weights)

inp = keras.Input(shape=(None, F), name="sequence_input")

valid_mask = layers.Lambda(
    lambda t: tf.reduce_any(tf.not_equal(t, PAD_X_VALUE), axis=-1),
    name="valid_mask",
)(inp)

x = layers.SpatialDropout1D(SPATIAL_DROPOUT)(inp)

x = layers.Bidirectional(
    layers.LSTM(
        LSTM_UNITS,
        return_sequences=True,
        dropout=LSTM_DROPOUT,
        recurrent_dropout=RECURRENT_DROPOUT,
    )
)(x, mask=valid_mask)

res_out = layers.Dense(1, activation="sigmoid")(x)
res_out = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="residue_out")(res_out)

model = keras.Model(inp, res_out)
model.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss=loss_fn,
)


def reduce_epoch_loss(batch_losses):
    total_loss = 0.0
    total_weight = 0.0

    for loss_value, batch_weight in batch_losses:
        total_loss += float(loss_value) * float(batch_weight)
        total_weight += float(batch_weight)

    if total_weight <= 0.0:
        raise RuntimeError("total batch weight must be positive")

    return total_loss / total_weight


def run_train_epoch(model, files, epoch_num):
    total_batches = int(np.ceil(len(files) / BATCH_SIZE))
    batch_losses = []

    print(f"starting train epoch {epoch_num}", flush=True)

    for step, batch_files in enumerate(
        iter_batch_file_lists(
            files,
            BATCH_SIZE,
            shuffle=True,
            seed=RANDOM_SEED + epoch_num,
        ),
        start=1,
    ):
        if step == 1:
            print(f"epoch {epoch_num} train batch 1/{total_batches} start", flush=True)

        Xb, yb, swb = make_padded_batch(batch_files)
        out = model.train_on_batch(
            Xb,
            yb,
            sample_weight=swb,
            return_dict=True,
        )
        loss_value = float(out["loss"])
        batch_weight = float(np.sum(swb))
        batch_losses.append((loss_value, batch_weight))

        print(
            f"epoch {epoch_num} train batch {step}/{total_batches} loss={loss_value:.4f}",
            end="\r",
            flush=True,
        )

    print(" " * 120, end="\r", flush=True)
    return reduce_epoch_loss(batch_losses)


def run_val_loss_epoch(model, files, epoch_num):
    total_batches = int(np.ceil(len(files) / BATCH_SIZE))
    batch_losses = []

    print(f"starting val epoch {epoch_num}", flush=True)

    for step, batch_files in enumerate(
        iter_batch_file_lists(
            files,
            BATCH_SIZE,
            shuffle=False,
        ),
        start=1,
    ):
        if step == 1:
            print(f"epoch {epoch_num} val batch 1/{total_batches} start", flush=True)

        Xb, yb, swb = make_padded_batch(batch_files)
        out = model.test_on_batch(
            Xb,
            yb,
            sample_weight=swb,
            return_dict=True,
        )
        loss_value = float(out["loss"])
        batch_weight = float(np.sum(swb))
        batch_losses.append((loss_value, batch_weight))

        print(
            f"epoch {epoch_num} val batch {step}/{total_batches} loss={loss_value:.4f}",
            end="\r",
            flush=True,
        )

    print(" " * 120, end="\r", flush=True)
    return reduce_epoch_loss(batch_losses)


def collect_stream_predictions_from_files(model, files, thr):
    y_true_chunks = []
    y_pred_chunks = []
    y_prob_chunks = []
    res_conf_chunks = []
    frag_conf_chunks = []

    for batch_files in iter_batch_file_lists(files, BATCH_SIZE, shuffle=False):
        Xb, yb, swb = make_padded_batch(batch_files)

        y_true = (yb >= 0.5).astype(np.int32)
        valid = (swb > 0.0).astype(np.float32)

        probs = model(Xb, training=False).numpy()
        y_pred = (probs > thr).astype(np.int32)

        valid_bool = valid > 0.0
        y_true_chunks.append(y_true[valid_bool])
        y_pred_chunks.append(y_pred[valid_bool])
        y_prob_chunks.append(probs[valid_bool])

        conf = np.maximum(probs, 1.0 - probs)
        res_conf_chunks.append(conf[valid_bool])

        lengths = np.maximum(valid.sum(axis=1), 1.0)
        frag_conf = (conf * valid).sum(axis=1) / lengths
        frag_conf_chunks.append(frag_conf)

    y_true_flat = np.concatenate(y_true_chunks, axis=0)
    y_pred_flat = np.concatenate(y_pred_chunks, axis=0)
    y_prob_flat = np.concatenate(y_prob_chunks, axis=0)
    res_conf_flat = np.concatenate(res_conf_chunks, axis=0)
    frag_conf = np.concatenate(frag_conf_chunks, axis=0)

    return y_true_flat, y_pred_flat, y_prob_flat, res_conf_flat, frag_conf


def compute_label_metrics(y_true_flat, y_pred_flat, f1_mode):
    y_true_flat = np.asarray(y_true_flat, dtype=np.int32)
    y_pred_flat = np.asarray(y_pred_flat, dtype=np.int32)

    precisions = []
    recalls = []

    for label in (0, 1):
        tp = int(np.sum((y_true_flat == label) & (y_pred_flat == label)))
        pred_pos = int(np.sum(y_pred_flat == label))
        true_pos = int(np.sum(y_true_flat == label))

        precision = (tp / pred_pos) if pred_pos > 0 else 0.0
        recall = (tp / true_pos) if true_pos > 0 else 0.0

        precisions.append(float(precision))
        recalls.append(float(recall))

    p = np.asarray(precisions, dtype=np.float32)
    r = np.asarray(recalls, dtype=np.float32)

    if f1_mode == "mean_pr":
        f = 0.5 * (p + r)
    elif f1_mode == "harmonic":
        denom = p + r
        f = np.where(denom > 0, (2.0 * p * r) / denom, 0.0)
    else:
        raise ValueError("F1_MODE must be 'mean_pr' or 'harmonic'")

    macro = float(np.mean(f))

    return {
        "macro_f1": float(macro),
        "f1_label0": float(f[0]),
        "f1_label1": float(f[1]),
        "precision_label0": float(p[0]),
        "precision_label1": float(p[1]),
        "recall_label0": float(r[0]),
        "recall_label1": float(r[1]),
    }


def eval_metrics_from_files(model, files, thr, f1_mode):
    yt, yp, _, _, _ = collect_stream_predictions_from_files(model, files, thr)
    return compute_label_metrics(yt, yp, f1_mode)


def fmt_metric(x):
    return "nan" if not np.isfinite(x) else f"{x:.4f}"


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
        if s <= (last_e + 1):
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

    if (e - s + 1) >= int(min_len):
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

        if remaining > 0 and e2 < (n_total - 1):
            shift_right = min(remaining, (n_total - 1) - e2)
            e2 += shift_right
            remaining -= shift_right

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

        run_len = j - i + 1
        if run_len >= int(min_zero_len):
            count += 1

        i = j + 1

    return count


def flatten_sequence_key(rows, key, dtype=np.int32):
    if not rows:
        return np.array([], dtype=dtype)
    return np.concatenate([np.asarray(row[key], dtype=dtype) for row in rows], axis=0)


def add_smoothed_predictions_to_rows(rows, min_len=min_residues_1s_smooth):
    for row in rows:
        row["pred_smoothed"] = smooth_label1_runs(row["pred"], min_len=min_len)
    return rows


def collect_domain_count_vectors(rows, pred_key):
    true_counts = []
    pred_counts = []

    for row in rows:
        true_counts.append(
            count_domains_from_labels(
                row["true"],
                min_zero_len=num_residues_0_domain_definition,
            )
        )
        pred_counts.append(
            count_domains_from_labels(
                row[pred_key],
                min_zero_len=num_residues_0_domain_definition,
            )
        )

    return np.asarray(true_counts, dtype=np.int32), np.asarray(pred_counts, dtype=np.int32)


def compute_domain_count_metrics(true_counts, pred_counts):
    true_counts = np.asarray(true_counts, dtype=np.int32)
    pred_counts = np.asarray(pred_counts, dtype=np.int32)

    if len(true_counts) == 0:
        raise RuntimeError("no sequences available for domain-count metrics")
    if len(true_counts) != len(pred_counts):
        raise RuntimeError("true_counts and pred_counts length mismatch")

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

    return {
        "n_sequences": n,
        "diff": diff,
        "counts": counts,
        "props": props,
        "mae": mae,
    }


def plot_domain_count_category_barplot(metrics_dict, title, out_path: Path):
    import matplotlib.pyplot as plt

    labels = [
        "Exact",
        "1 too few",
        "1 too many",
        "2 too few",
        "2 too many",
        "3+ too few",
        "3+ too many",
    ]
    counts = [
        metrics_dict["counts"]["exact"],
        metrics_dict["counts"]["one_too_few"],
        metrics_dict["counts"]["one_too_many"],
        metrics_dict["counts"]["two_too_few"],
        metrics_dict["counts"]["two_too_many"],
        metrics_dict["counts"]["three_or_more_too_few"],
        metrics_dict["counts"]["three_or_more_too_many"],
    ]

    colors = [
        "#000000",
        "#4c78a8",
        "#f58518",
        "#54a24b",
        "#e45756",
        "#72b7b2",
        "#b279a2",
    ]

    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.6)

    ymax = max(counts) if counts else 0
    plt.ylim(0, max(1, ymax) * 1.15 + 1)

    for bar, val in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.2,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.ylabel("Number of Test Sequences")
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
    n_take = min(n_sequences, len(rows))
    selected = rows[:n_take]

    pred_before_rows = []
    pred_after_rows = []
    true_rows = []

    for row in selected:
        pred_before_rows.append(row["pred"])
        pred_after_rows.append(row["pred_smoothed"])
        true_rows.append(row["true"])

    pred_before_mat = pad_rows(pred_before_rows, pad_value=-1, dtype=np.int8)
    pred_after_mat = pad_rows(pred_after_rows, pad_value=-1, dtype=np.int8)
    true_mat = pad_rows(true_rows, pad_value=-1, dtype=np.int8)
    return pred_before_mat, pred_after_mat, true_mat


def plot_triptych(pred_before_bin, pred_after_bin, true_bin, out_path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    binary_cmap = ListedColormap(["#d0d0d0", "#000000", "#ff69b4"])
    binary_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], binary_cmap.N)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, max(6, pred_before_bin.shape[0] * 0.12)),
        sharey=True,
        constrained_layout=True,
    )

    axes[0].imshow(
        pred_before_bin,
        aspect="auto",
        interpolation="none",
        cmap=binary_cmap,
        norm=binary_norm,
    )
    axes[0].set_title("Predicted Labels Before Smoothing")
    axes[0].set_xlabel("Residue Index")
    axes[0].set_ylabel("Test Sequence #")

    axes[1].imshow(
        pred_after_bin,
        aspect="auto",
        interpolation="none",
        cmap=binary_cmap,
        norm=binary_norm,
    )
    axes[1].set_title("Predicted Labels After Smoothing")
    axes[1].set_xlabel("Residue Index")

    axes[2].imshow(
        true_bin,
        aspect="auto",
        interpolation="none",
        cmap=binary_cmap,
        norm=binary_norm,
    )
    axes[2].set_title("True Labels")
    axes[2].set_xlabel("Residue Index")

    fig.suptitle(
        f"Test Sequences (first {pred_before_bin.shape[0]}): "
        f"Pred Before vs Pred After vs True"
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def threshold_folder_name(thr):
    s = f"{thr:.3f}".rstrip("0").rstrip(".")
    return f"pred{s}"


def predict_files_in_batches(model, files):
    rows = []
    for start in range(0, len(files), BATCH_SIZE):
        chunk = files[start:start + BATCH_SIZE]
        lengths = []
        X_list = []
        y_list = []

        for f in chunk:
            X, y = load_npz_raw(f)
            lengths.append(int(X.shape[0]))
            X_list.append(X)
            y_list.append((y >= 0.5).astype(np.int8))

        max_len = max(lengths)
        X_batch = np.full((len(chunk), max_len, F), PAD_X_VALUE, dtype=np.float32)
        y_batch = np.zeros((len(chunk), max_len), dtype=np.int8)

        for i, (X, y, seq_len) in enumerate(zip(X_list, y_list, lengths)):
            X_batch[i, :seq_len, :] = X
            y_batch[i, :seq_len] = y

        prob_np = model(X_batch, training=False).numpy()

        for i, f in enumerate(chunk):
            seq_len = lengths[i]
            rows.append(
                {
                    "file": f,
                    "entry_id": f.stem,
                    "length": seq_len,
                    "true": y_batch[i, :seq_len].copy(),
                    "prob": prob_np[i, :seq_len].copy(),
                }
            )

    return rows


def threshold_prediction_rows(rows, thr):
    out = []
    for row in rows:
        out.append(
            {
                "file": row["file"],
                "entry_id": row["entry_id"],
                "length": row["length"],
                "true": row["true"],
                "prob": row["prob"],
                "pred": (row["prob"] > thr).astype(np.int8),
            }
        )
    return out


def compute_confidence_vectors_from_prob_rows(rows):
    res_conf_chunks = []
    frag_conf_chunks = []

    for row in rows:
        conf = np.maximum(row["prob"], 1.0 - row["prob"])
        res_conf_chunks.append(conf)
        frag_conf_chunks.append(float(conf.mean()))

    if not res_conf_chunks:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    return (
        np.concatenate(res_conf_chunks, axis=0),
        np.asarray(frag_conf_chunks, dtype=np.float32),
    )


def collect_flat_from_rows(rows, pred_key="pred"):
    y_true_flat = flatten_sequence_key(rows, "true", dtype=np.int32)
    y_pred_flat = flatten_sequence_key(rows, pred_key, dtype=np.int32)
    y_prob_flat = flatten_sequence_key(rows, "prob", dtype=np.float32)
    return y_true_flat, y_pred_flat, y_prob_flat


def labels_to_bitstring(arr):
    return "".join("1" if int(x) == 1 else "0" for x in arr.tolist())


def pick_col(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        k = cand.lower().strip()
        if k in lower:
            return lower[k]
    for c in df.columns:
        cl = str(c).strip().lower()
        for cand in candidates:
            if cand.lower().strip() in cl:
                return c
    return None


def normalize_entry_id(x):
    return str(x).strip().upper()


def normalize_sequence(x):
    return "".join(str(x).strip().upper().split())


def load_targ_sequence_lookup(csv_path: Path):
    import pandas as pd

    if not csv_path.exists():
        raise RuntimeError(f"missing targ_deduped csv: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    if df.empty:
        raise RuntimeError(f"empty targ_deduped csv: {csv_path}")

    entry_col = pick_col(df, ["Entry ID", "entry_id", "Entry", "ID"])
    seq_col = pick_col(df, ["Sequence", "sequence"])
    if entry_col is None or seq_col is None:
        raise RuntimeError(
            f"could not find entry id / sequence columns in {csv_path}. found: {df.columns.tolist()}"
        )

    out = {}
    for _, row in df.iterrows():
        entry_raw = str(row[entry_col]).strip()
        seq = normalize_sequence(row[seq_col])
        if not entry_raw or not seq:
            continue
        key = normalize_entry_id(entry_raw)
        if key in out:
            raise RuntimeError(f"duplicate entry id in {csv_path}: {entry_raw}")
        out[key] = {"Entry ID": entry_raw, "Sequence": seq}

    if not out:
        raise RuntimeError(f"no usable entry id / sequence rows in {csv_path}")
    return out


def export_full_val_sequence_predictions_from_rows(pred_rows, targ_csv: Path, out_csv: Path):
    import pandas as pd

    lookup = load_targ_sequence_lookup(targ_csv)

    if not pred_rows:
        raise RuntimeError("no full-val predictions were collected")

    rows = []
    for row in pred_rows:
        key = normalize_entry_id(row["entry_id"])
        if key not in lookup:
            raise RuntimeError(f"entry id from npz files not found in {targ_csv}: {row['entry_id']}")

        meta = lookup[key]
        seq = meta["Sequence"]
        entry_out = meta["Entry ID"]

        pred_labels = labels_to_bitstring(row["pred"])
        covered_len = min(len(pred_labels), len(seq))
        if covered_len <= 0:
            raise RuntimeError(f"no overlapping covered residues for {entry_out}")

        rows.append(
            {
                "Entry ID": entry_out,
                "Sequence": seq[:covered_len],
                "pred_labels": pred_labels[:covered_len],
            }
        )

    out_df = pd.DataFrame(rows, columns=["Entry ID", "Sequence", "pred_labels"])
    out_df = out_df.sort_values("Entry ID", kind="mergesort").reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)
    return out_df


def plot_conf_hist(vals, title, fname):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(vals, bins=50, range=(0, 1), color="#ff4da6", alpha=0.85)
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=300)
    plt.close()


def save_history_csv(train_losses, val_losses, macro_f1s, f1_0s, f1_1s):
    import pandas as pd

    pd.DataFrame(
        {
            "epoch": np.arange(1, len(train_losses) + 1),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_macro_f1": macro_f1s,
            "val_f1_0": f1_0s,
            "val_f1_1": f1_1s,
        }
    ).to_csv(OUT_DIR / "history.csv", index=False)


def save_curve_plots(train_losses, val_losses, macro_f1s, f1_0s, f1_1s):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curves.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(macro_f1s, label="VAL Macro F1")
    plt.plot(f1_0s, label="VAL Label 0 F1")
    plt.plot(f1_1s, label="VAL Label 1 F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "f1_curves.png", dpi=300)
    plt.close()


train_losses, val_losses = [], []
macro_f1s, f1_0s, f1_1s = [], [], []

best_epoch = -1
best_weights = None
no_improve = 0

if EARLY_STOP_VAL_MACRO:
    best_monitor = -np.inf
    monitor_name = "val_macro_f1"
else:
    best_monitor = np.inf
    monitor_name = "val_loss"

print(f"training with early-stop monitor: {monitor_name}; f1_mode={F1_MODE}")
for epoch in range(EPOCHS):
    epoch_num = epoch + 1
    print(f"starting epoch {epoch_num}", flush=True)

    tr_loss = run_train_epoch(model, tr_files, epoch_num)
    va_loss = run_val_loss_epoch(model, val_files, epoch_num)

    train_losses.append(tr_loss)
    val_losses.append(va_loss)

    val_metrics = eval_metrics_from_files(model, val_files, PRED_THRESHOLD, F1_MODE)
    macro = val_metrics["macro_f1"]
    f10 = val_metrics["f1_label0"]
    f11 = val_metrics["f1_label1"]

    macro_f1s.append(macro)
    f1_0s.append(f10)
    f1_1s.append(f11)

    print(
        f"epoch {epoch_num}: "
        f"train_loss={tr_loss:.4f} | "
        f"val_loss={va_loss:.4f} | "
        f"val_macro={macro:.4f} | "
        f"val_f10={f10:.4f} | "
        f"val_f11={f11:.4f}",
        flush=True,
    )

    if EARLY_STOP_VAL_MACRO:
        improved = macro > best_monitor
        if improved:
            best_monitor = macro
            best_epoch = epoch_num
            best_weights = model.get_weights()
            no_improve = 0
        else:
            no_improve += 1
    else:
        improved = va_loss < best_monitor
        if improved:
            best_monitor = va_loss
            best_epoch = epoch_num
            best_weights = model.get_weights()
            no_improve = 0
        else:
            no_improve += 1

    if no_improve >= EARLY_STOP_PATIENCE:
        print(f"early stopping at epoch {epoch_num}")
        break

if best_weights is None:
    best_epoch = len(train_losses)
    best_weights = model.get_weights()

print(f"restoring best model from epoch {best_epoch}")
model.set_weights(best_weights)

val_prob_rows = predict_files_in_batches(model, val_files)
test_prob_rows = predict_files_in_batches(model, test_files)
all_prob_rows = predict_files_in_batches(model, all_files)

test_res_conf, test_frag_conf = compute_confidence_vectors_from_prob_rows(test_prob_rows)

created_pred_dirs = []

for thr in THRESHOLD_SWEEP:
    pred_dir = OUT_DIR / threshold_folder_name(thr)
    pred_dir.mkdir(parents=True, exist_ok=True)
    created_pred_dirs.append(pred_dir)

    val_rows_thr = threshold_prediction_rows(val_prob_rows, thr)
    test_rows_thr = threshold_prediction_rows(test_prob_rows, thr)
    all_rows_thr = threshold_prediction_rows(all_prob_rows, thr)

    val_yt, val_yp, _ = collect_flat_from_rows(val_rows_thr, pred_key="pred")
    test_yt, test_yp, _ = collect_flat_from_rows(test_rows_thr, pred_key="pred")

    val_metrics = compute_label_metrics(val_yt, val_yp, F1_MODE)
    test_metrics = compute_label_metrics(test_yt, test_yp, F1_MODE)

    test_rows_thr_smoothed = add_smoothed_predictions_to_rows(
        test_rows_thr,
        min_len=min_residues_1s_smooth,
    )
    test_yt_smoothed = flatten_sequence_key(test_rows_thr_smoothed, "true", dtype=np.int32)
    test_yp_smoothed = flatten_sequence_key(test_rows_thr_smoothed, "pred_smoothed", dtype=np.int32)
    test_metrics_smoothed = compute_label_metrics(test_yt_smoothed, test_yp_smoothed, F1_MODE)

    domain_true_before, domain_pred_before = collect_domain_count_vectors(test_rows_thr, pred_key="pred")
    domain_true_after, domain_pred_after = collect_domain_count_vectors(test_rows_thr_smoothed, pred_key="pred_smoothed")

    domain_count_metrics_before = compute_domain_count_metrics(domain_true_before, domain_pred_before)
    domain_count_metrics_after = compute_domain_count_metrics(domain_true_after, domain_pred_after)

    plot_domain_count_category_barplot(
        domain_count_metrics_before,
        f"Test Domain-Count Error Categories (Before Smoothing, thr={thr})",
        pred_dir / DOMAIN_COUNT_BARPLOT_BEFORE_FILE,
    )
    plot_domain_count_category_barplot(
        domain_count_metrics_after,
        f"Test Domain-Count Error Categories (After Smoothing, thr={thr})",
        pred_dir / DOMAIN_COUNT_BARPLOT_AFTER_FILE,
    )

    pred_before_n, pred_after_n, true_bin_n = collect_first_n_test_triptych(
        test_rows_thr_smoothed,
        TEST_HEATMAP_N,
    )
    plot_triptych(pred_before_n, pred_after_n, true_bin_n, pred_dir / TEST_HEATMAP_FILE)

    full_val_pred_df = export_full_val_sequence_predictions_from_rows(
        pred_rows=all_rows_thr,
        targ_csv=TARG_DEDUPED_CSV,
        out_csv=pred_dir / FULL_VAL_SEQUENCE_PRED_CSV,
    )

    (pred_dir / "final_summary.txt").write_text(
        f"BEST EPOCH: {best_epoch}\n"
        f"EARLY_STOP_MODE: {'val_macro_f1' if EARLY_STOP_VAL_MACRO else 'val_loss'}\n"
        f"F1_MODE: {F1_MODE}\n"
        f"PRED_THRESHOLD: {thr}\n"
        f"VAL  Macro F1: {val_metrics['macro_f1']:.4f}\n"
        f"VAL  F1 label0: {val_metrics['f1_label0']:.4f}\n"
        f"VAL  F1 label1: {val_metrics['f1_label1']:.4f}\n"
        f"TEST BEFORE SMOOTHING Macro F1: {test_metrics['macro_f1']:.4f}\n"
        f"TEST BEFORE SMOOTHING F1 label0: {test_metrics['f1_label0']:.4f}\n"
        f"TEST BEFORE SMOOTHING F1 label1: {test_metrics['f1_label1']:.4f}\n"
        f"TEST BEFORE SMOOTHING Precision label1: {test_metrics['precision_label1']:.4f}\n"
        f"TEST BEFORE SMOOTHING Recall label1: {test_metrics['recall_label1']:.4f}\n"
        f"TEST AFTER SMOOTHING Macro F1: {test_metrics_smoothed['macro_f1']:.4f}\n"
        f"TEST AFTER SMOOTHING F1 label0: {test_metrics_smoothed['f1_label0']:.4f}\n"
        f"TEST AFTER SMOOTHING F1 label1: {test_metrics_smoothed['f1_label1']:.4f}\n"
        f"TEST AFTER SMOOTHING Precision label1: {test_metrics_smoothed['precision_label1']:.4f}\n"
        f"TEST AFTER SMOOTHING Recall label1: {test_metrics_smoothed['recall_label1']:.4f}\n"
        f"TEST Domain-count exact proportion before smoothing: {fmt_metric(domain_count_metrics_before['props']['exact'])}\n"
        f"TEST Domain-count one-too-few proportion before smoothing: {fmt_metric(domain_count_metrics_before['props']['one_too_few'])}\n"
        f"TEST Domain-count one-too-many proportion before smoothing: {fmt_metric(domain_count_metrics_before['props']['one_too_many'])}\n"
        f"TEST Domain-count two-too-few proportion before smoothing: {fmt_metric(domain_count_metrics_before['props']['two_too_few'])}\n"
        f"TEST Domain-count two-too-many proportion before smoothing: {fmt_metric(domain_count_metrics_before['props']['two_too_many'])}\n"
        f"TEST Domain-count three-or-more-too-few proportion before smoothing: {fmt_metric(domain_count_metrics_before['props']['three_or_more_too_few'])}\n"
        f"TEST Domain-count three-or-more-too-many proportion before smoothing: {fmt_metric(domain_count_metrics_before['props']['three_or_more_too_many'])}\n"
        f"TEST Domain-count MAE before smoothing: {fmt_metric(domain_count_metrics_before['mae'])}\n"
        f"TEST Domain-count exact proportion after smoothing: {fmt_metric(domain_count_metrics_after['props']['exact'])}\n"
        f"TEST Domain-count one-too-few proportion after smoothing: {fmt_metric(domain_count_metrics_after['props']['one_too_few'])}\n"
        f"TEST Domain-count one-too-many proportion after smoothing: {fmt_metric(domain_count_metrics_after['props']['one_too_many'])}\n"
        f"TEST Domain-count two-too-few proportion after smoothing: {fmt_metric(domain_count_metrics_after['props']['two_too_few'])}\n"
        f"TEST Domain-count two-too-many proportion after smoothing: {fmt_metric(domain_count_metrics_after['props']['two_too_many'])}\n"
        f"TEST Domain-count three-or-more-too-few proportion after smoothing: {fmt_metric(domain_count_metrics_after['props']['three_or_more_too_few'])}\n"
        f"TEST Domain-count three-or-more-too-many proportion after smoothing: {fmt_metric(domain_count_metrics_after['props']['three_or_more_too_many'])}\n"
        f"TEST Domain-count MAE after smoothing: {fmt_metric(domain_count_metrics_after['mae'])}\n"
        f"DOMAIN COUNT RULE: number of label-0 stretches with length >= {num_residues_0_domain_definition}\n"
        f"BOUNDARY SMOOTHING RULE: predicted label-1 runs are merged if the zero-gap between them <= {min_residues_1s_smooth}, "
        f"and any resulting label-1 run shorter than {min_residues_1s_smooth} is expanded outward to length {min_residues_1s_smooth}\n",
        encoding="utf-8",
    )

plot_conf_hist(test_res_conf, "Test Residue Confidence Distribution", "dist_test_residue_conf.png")
plot_conf_hist(test_frag_conf, "Test Sequence Confidence Distribution", "dist_test_fragment_conf.png")

save_history_csv(train_losses, val_losses, macro_f1s, f1_0s, f1_1s)
save_curve_plots(train_losses, val_losses, macro_f1s, f1_0s, f1_1s)

model.save(OUT_DIR / "model_best.keras")

print(f"saved results to {OUT_DIR}")
print("done")
