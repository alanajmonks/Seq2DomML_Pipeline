import argparse
import csv
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


FULL_DATASET_DIR = Path("data/full_dataset_with_features")
TRAIN_DIR = Path("data/split_dataset/train")
TEST_AND_VAL_DIR = Path("data/split_dataset/test_and_val")
OUT_ROOT = Path("data/pruned_featuresets")
BARPLOT_FILE = OUT_ROOT / "features_correlation_barplot.png"
FEAT_NAMES_FILE = Path("data/split_dataset/feat_names")

THRESHOLD_LABELS = [
    "0.025",
    "0.05",
    "0.1",
    "0.15",
    "0.2",
    "0.25",
    "0.3",
    "0.4",
    "0.5",
    "0.6",
    "0.7",
    "0.8",
    "0.9",
    "1.0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dataset-dir", type=Path, default=FULL_DATASET_DIR)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--test-and-val-dir", type=Path, default=TEST_AND_VAL_DIR)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--barplot-file", type=Path, default=BARPLOT_FILE)
    parser.add_argument("--feat-names-file", type=Path, default=FEAT_NAMES_FILE)
    parser.add_argument("--threshold-labels", nargs="+", default=THRESHOLD_LABELS)
    return parser.parse_args()


def list_npz_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.npz"))
    if not files:
        raise SystemExit(1)
    return files


def normalize_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_feature_names(npz_path: Path, feat_names_file: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as z:
        X = z["X"]
        f = int(X.shape[1])

        if "feature_names" in z.files:
            names = np.asarray([normalize_name(v) for v in z["feature_names"].reshape(-1)], dtype=object)
            if len(names) == f:
                return names

        if "feat_names" in z.files:
            names = np.asarray([normalize_name(v) for v in z["feat_names"].reshape(-1)], dtype=object)
            if len(names) == f:
                return names

    if feat_names_file.exists():
        rows = []
        for line in feat_names_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            idx_txt, name = parts
            rows.append((int(idx_txt), name))
        rows.sort(key=lambda x: x[0])
        names = np.asarray([name for _, name in rows], dtype=object)
        if len(names) == f:
            return names

    return np.asarray([f"feature_{i}" for i in range(f)], dtype=object)


def parse_thresholds(labels: list[str]) -> list[tuple[str, float]]:
    out = []
    for label in labels:
        value = float(label)
        if value <= 0.0 or value > 1.0:
            raise SystemExit(1)
        out.append((label, value))
    return out


def compute_correlations(files: list[Path]) -> np.ndarray:
    with np.load(files[0], allow_pickle=True) as z:
        f = int(z["X"].shape[1])

    sum_x = np.zeros(f, dtype=np.float64)
    sum_x2 = np.zeros(f, dtype=np.float64)
    sum_xy = np.zeros(f, dtype=np.float64)
    sum_y = 0.0
    n = 0

    print("computing correlations from train set only...")
    with tqdm(total=len(files)) as pbar:
        for path in files:
            with np.load(path, allow_pickle=True) as z:
                X = z["X"].astype(np.float64, copy=False)
                y = (z["y"] >= 0.5).astype(np.float64, copy=False)

            sum_x += X.sum(axis=0)
            sum_x2 += np.square(X).sum(axis=0)
            sum_xy += (X * y[:, None]).sum(axis=0)
            sum_y += y.sum()
            n += len(y)
            pbar.update(1)

    if n == 0:
        raise SystemExit(1)

    mean_x = sum_x / n
    mean_y = sum_y / n
    var_x = (sum_x2 / n) - np.square(mean_x)
    var_x = np.maximum(var_x, 0.0)
    var_y = mean_y * (1.0 - mean_y)

    corr = np.zeros(f, dtype=np.float64)
    if var_y > 0.0:
        cov = (sum_xy / n) - (mean_x * mean_y)
        denom = np.sqrt(var_x * var_y)
        mask = denom > 0.0
        corr[mask] = cov[mask] / denom[mask]

    return np.abs(corr)


def compute_rank_desc(correlations: np.ndarray) -> np.ndarray:
    order = np.argsort(-correlations, kind="mergesort")
    rank_desc = np.empty(len(correlations), dtype=np.int64)
    rank_desc[order] = np.arange(1, len(correlations) + 1, dtype=np.int64)
    return rank_desc


def save_barplot(correlations: np.ndarray, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ordered = np.sort(correlations)[::-1]
    x = np.arange(len(ordered))

    plt.figure(figsize=(16, 6))
    plt.bar(x, ordered, width=1.0)
    plt.xlabel("Features ordered from most to least correlated")
    plt.ylabel("Absolute point-biserial correlation")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def prepare_output_tree(out_root: Path, thresholds: list[tuple[str, float]]) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for label, _ in thresholds:
        threshold_dir = out_root / label
        (threshold_dir / "train").mkdir(parents=True, exist_ok=True)
        (threshold_dir / "test_and_val").mkdir(parents=True, exist_ok=True)


def select_feature_indices(correlations: np.ndarray, thresholds: list[tuple[str, float]]) -> dict[str, np.ndarray]:
    rank_order = np.argsort(-correlations, kind="mergesort")
    f = len(correlations)
    selected = {}

    for label, pct in thresholds:
        if pct >= 1.0:
            k = f
        else:
            k = int(math.floor(f * pct))
            k = max(1, min(k, f))
        keep = np.sort(rank_order[:k])
        selected[label] = keep

    return selected


def write_feature_manifest(
    out_path: Path,
    keep_idx: np.ndarray,
    feature_names: np.ndarray,
    correlations: np.ndarray,
    rank_desc: np.ndarray,
) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pruned_feature_idx", "original_feature_idx", "feature_name", "abs_corr", "rank_desc"])
        for pruned_idx, original_idx in enumerate(keep_idx.tolist()):
            writer.writerow(
                [
                    pruned_idx,
                    int(original_idx),
                    feature_names[original_idx],
                    float(correlations[original_idx]),
                    int(rank_desc[original_idx]),
                ]
            )


def write_pruned_npz(
    src_path: Path,
    dst_path: Path,
    keep_idx: np.ndarray,
    feature_names: np.ndarray,
) -> None:
    with np.load(src_path, allow_pickle=True) as z:
        X = z["X"][:, keep_idx]
        y = z["y"]

    np.savez_compressed(
        dst_path,
        X=X,
        y=y,
        feature_names=feature_names[keep_idx],
        original_feature_indices=keep_idx.astype(np.int32, copy=False),
    )


def write_threshold_dataset(
    label: str,
    keep_idx: np.ndarray,
    train_files: list[Path],
    test_and_val_files: list[Path],
    out_root: Path,
    feature_names: np.ndarray,
    correlations: np.ndarray,
    rank_desc: np.ndarray,
) -> None:
    threshold_dir = out_root / label
    manifest_path = threshold_dir / "feature_manifest.csv"

    print(f"writing manifest for threshold {label}...")
    write_feature_manifest(
        out_path=manifest_path,
        keep_idx=keep_idx,
        feature_names=feature_names,
        correlations=correlations,
        rank_desc=rank_desc,
    )

    print(f"outputting pruned dataset with {label} most correlated features...")
    total = len(train_files) + len(test_and_val_files)

    with tqdm(total=total) as pbar:
        train_out = threshold_dir / "train"
        for path in train_files:
            write_pruned_npz(path, train_out / path.name, keep_idx, feature_names)
            pbar.update(1)

        test_and_val_out = threshold_dir / "test_and_val"
        for path in test_and_val_files:
            write_pruned_npz(path, test_and_val_out / path.name, keep_idx, feature_names)
            pbar.update(1)


def main() -> None:
    args = parse_args()

    train_files = list_npz_files(args.train_dir)
    test_and_val_files = list_npz_files(args.test_and_val_dir)

    thresholds = parse_thresholds(args.threshold_labels)
    correlations = compute_correlations(train_files)
    rank_desc = compute_rank_desc(correlations)
    feature_names = load_feature_names(train_files[0], args.feat_names_file)

    prepare_output_tree(args.out_root, thresholds)
    save_barplot(correlations, args.barplot_file)

    selected = select_feature_indices(correlations, thresholds)

    for label, _ in thresholds:
        write_threshold_dataset(
            label=label,
            keep_idx=selected[label],
            train_files=train_files,
            test_and_val_files=test_and_val_files,
            out_root=args.out_root,
            feature_names=feature_names,
            correlations=correlations,
            rank_desc=rank_desc,
        )


if __name__ == "__main__":
    main()
