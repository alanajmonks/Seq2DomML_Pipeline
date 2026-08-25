import argparse
import ast
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from tqdm import tqdm


INPUT_CSV = Path("data/preprocessing/targ_deduped.csv")
OUTPUT_DIR = Path("data/full_dataset_with_features")
INDICES_TXT = Path("features/aa_indices/indices.txt")
LOOKUP_TABLES_TXT = Path("features/aa_indices/lookup_tables.txt")
MATRIX_DIR = Path("features/interaction_matrices")
SCALER_DIR = Path("features/scalers")

ENTRY_COL = "Entry ID"
SEQ_COL = "Sequence"
TARGET_COL = "targ_binary"

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA)
MATRIX_AA = list("ARNDCQEGHILKMFPSTWYV")
TMP_DIR_NAME = ".tmp_feature_parts"


@dataclass(frozen=True)
class Record:
    entry_id: str
    sequence: str
    target: np.ndarray
    stem: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--indices-txt", type=Path, default=INDICES_TXT)
    parser.add_argument("--lookup-tables-txt", type=Path, default=LOOKUP_TABLES_TXT)
    parser.add_argument("--matrix-dir", type=Path, default=MATRIX_DIR)
    parser.add_argument("--scaler-dir", type=Path, default=SCALER_DIR)
    return parser.parse_args()


def clean_stem(entry_id: str) -> str:
    return entry_id.replace("/", "_").replace("\\", "_").strip()


def load_records(path: Path) -> list[Record]:
    if not path.exists():
        raise SystemExit(1)

    df = pd.read_csv(path, dtype=str)
    if ENTRY_COL not in df.columns or SEQ_COL not in df.columns or TARGET_COL not in df.columns:
        raise SystemExit(1)

    records: list[Record] = []
    seen_entry_ids: set[str] = set()
    seen_stems: set[str] = set()

    for _, row in df.iterrows():
        entry_id = str(row[ENTRY_COL]).strip()
        sequence = str(row[SEQ_COL]).strip().upper()
        target_str = "".join(ch for ch in str(row[TARGET_COL]) if ch in {"0", "1"})

        if not entry_id or not sequence or not target_str:
            continue

        if any(aa not in AA_SET for aa in sequence):
            continue

        length = min(len(sequence), len(target_str))
        if length == 0:
            continue

        if entry_id in seen_entry_ids:
            raise SystemExit(1)

        stem = clean_stem(entry_id)
        if not stem or stem in seen_stems:
            raise SystemExit(1)

        seen_entry_ids.add(entry_id)
        seen_stems.add(stem)

        sequence = sequence[:length]
        target = np.fromiter((int(ch) for ch in target_str[:length]), dtype=np.int8, count=length)

        records.append(
            Record(
                entry_id=entry_id,
                sequence=sequence,
                target=target,
                stem=stem,
            )
        )

    return records


def load_lookup_tables(indices_txt: Path, lookup_tables_txt: Path) -> list[tuple[str, dict]]:
    if not indices_txt.exists() or not lookup_tables_txt.exists():
        raise SystemExit(1)

    source = indices_txt.read_text(encoding="utf-8") + "\n" + lookup_tables_txt.read_text(encoding="utf-8")
    module = ast.parse(source)

    tables: dict[str, dict] = {}
    lookup_refs: list[tuple[str, str]] = []

    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue

        name = node.targets[0].id

        if name == "lookup_tables" and isinstance(node.value, ast.List):
            for elt in node.value.elts:
                if not isinstance(elt, ast.Tuple) or len(elt.elts) != 2:
                    continue
                if not isinstance(elt.elts[0], ast.Constant):
                    continue
                if not isinstance(elt.elts[1], ast.Name):
                    continue
                lookup_refs.append((str(elt.elts[0].value), elt.elts[1].id))
        elif isinstance(node.value, ast.Dict):
            tables[name] = ast.literal_eval(node.value)

    if not lookup_refs:
        raise SystemExit(1)

    out: list[tuple[str, dict]] = []
    for label, ref in lookup_refs:
        table = tables.get(ref)
        if table is None:
            raise SystemExit(1)
        out.append((label, table))

    return out


def classify_lookup_tables(lookup_tables: list[tuple[str, dict]]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if not lookup_tables:
        return (
            np.empty((len(AA), 0), dtype=np.float32),
            np.empty((len(AA), 0), dtype=np.float32),
            [],
            [],
        )

    table_values = []
    feature_names = []

    for name, table in lookup_tables:
        feature_names.append(f"pc_{name}")
        table_values.append(np.array([float(table.get(aa, np.nan)) for aa in AA], dtype=np.float32))

    stacked = np.vstack(table_values)

    binary_idx = []
    continuous_idx = []

    for idx in range(stacked.shape[0]):
        finite = stacked[idx][np.isfinite(stacked[idx])]
        uniq = set(float(x) for x in np.unique(finite).tolist())
        if len(uniq) <= 2 and uniq.issubset({0.0, 1.0}):
            binary_idx.append(idx)
        else:
            continuous_idx.append(idx)

    binary_values = stacked[binary_idx].T.astype(np.float32, copy=False) if binary_idx else np.empty((len(AA), 0), dtype=np.float32)
    continuous_values = stacked[continuous_idx].T.astype(np.float32, copy=False) if continuous_idx else np.empty((len(AA), 0), dtype=np.float32)
    binary_names = [feature_names[idx] for idx in binary_idx]
    continuous_names = [feature_names[idx] for idx in continuous_idx]

    return binary_values, continuous_values, binary_names, continuous_names


def load_matrices(matrix_dir: Path) -> tuple[np.ndarray, list[str]]:
    if not matrix_dir.exists():
        raise SystemExit(1)

    matrix_paths = sorted(matrix_dir.glob("*.csv"))
    if not matrix_paths:
        raise SystemExit(1)

    matrix_blocks = []
    matrix_feature_names = []

    for matrix_path in matrix_paths:
        df = pd.read_csv(matrix_path, index_col=0)
        df.index = df.index.astype(str).str.upper()
        df.columns = df.columns.astype(str).str.upper()

        if df.shape != (20, 20):
            raise SystemExit(1)
        if set(df.index) != set(MATRIX_AA) or set(df.columns) != set(MATRIX_AA):
            raise SystemExit(1)

        df = df.reindex(index=MATRIX_AA, columns=MATRIX_AA)
        matrix_blocks.append(df.to_numpy(dtype=np.float32))
        matrix_feature_names.extend([f"{aa}_{matrix_path.stem}" for aa in MATRIX_AA])

    return np.concatenate(matrix_blocks, axis=1).astype(np.float32, copy=False), matrix_feature_names


def prepare_dirs(output_dir: Path, scaler_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    for npz_path in output_dir.glob("*.npz"):
        npz_path.unlink()

    tmp_root = output_dir / TMP_DIR_NAME
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    paths = {
        "tmp_root": tmp_root,
        "bins": tmp_root / "bins",
        "pos": tmp_root / "pos",
        "aa_bin": tmp_root / "aa_bin",
        "aa_cont": tmp_root / "aa_cont",
        "matrix": tmp_root / "matrix",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def save_array(path: Path, array: np.ndarray) -> None:
    np.save(path, array)


def load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def sequence_to_indices(sequence: str, mapping: dict[str, int]) -> np.ndarray:
    return np.fromiter((mapping[aa] for aa in sequence), dtype=np.int16, count=len(sequence))


def positional_features(length: int) -> np.ndarray:
    idx0 = np.arange(length, dtype=np.float32)

    if length > 1:
        rel_pos = idx0 / np.float32(length - 1)
    else:
        rel_pos = np.zeros(length, dtype=np.float32)

    dist_to_term = np.minimum(idx0, np.float32(length - 1) - idx0) / np.float32(max(length / 2.0, 1.0))
    sinusoidal = np.sin((2.0 * np.pi * idx0) / np.float32(max(length, 1))).astype(np.float32)

    return np.column_stack((rel_pos.astype(np.float32), dist_to_term.astype(np.float32), sinusoidal))


def init_stats(n_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(n_features, dtype=np.float64),
        np.zeros(n_features, dtype=np.float64),
        np.zeros(n_features, dtype=np.int64),
    )


def update_stats(sum_x: np.ndarray, sum_x2: np.ndarray, count: np.ndarray, values: np.ndarray) -> None:
    if values.size == 0:
        return
    valid = np.isfinite(values)
    safe = np.where(valid, values, 0.0).astype(np.float64, copy=False)
    sum_x += safe.sum(axis=0)
    sum_x2 += np.square(safe).sum(axis=0)
    count += valid.sum(axis=0)


def finish_stats(sum_x: np.ndarray, sum_x2: np.ndarray, count: np.ndarray, zero_fill: float) -> tuple[np.ndarray, np.ndarray]:
    if sum_x.size == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    means = np.zeros(sum_x.shape[0], dtype=np.float64)
    stds = np.ones(sum_x.shape[0], dtype=np.float64)

    ok = count > 0
    means[ok] = sum_x[ok] / count[ok]

    variances = np.zeros(sum_x.shape[0], dtype=np.float64)
    variances[ok] = (sum_x2[ok] / count[ok]) - (means[ok] ** 2)
    variances = np.maximum(variances, 0.0)

    stds[ok] = np.sqrt(variances[ok])
    stds[stds == 0.0] = zero_fill

    return means.astype(np.float32), stds.astype(np.float32)


def encode_amino_acid_bins(records: list[Record], bins_dir: Path) -> None:
    eye = np.eye(len(AA), dtype=np.float32)
    aa_to_idx = {aa: idx for idx, aa in enumerate(AA)}

    for record in tqdm(records, total=len(records)):
        aa_idx = sequence_to_indices(record.sequence, aa_to_idx)
        save_array(bins_dir / f"{record.stem}.npy", eye[aa_idx])


def encode_positional_features(records: list[Record], pos_dir: Path) -> None:
    for record in tqdm(records, total=len(records)):
        save_array(pos_dir / f"{record.stem}.npy", positional_features(len(record.sequence)))


def encode_amino_acid_indices(
    records: list[Record],
    aa_binary_values: np.ndarray,
    aa_continuous_values: np.ndarray,
    aa_continuous_names: list[str],
    aa_bin_dir: Path,
    aa_cont_dir: Path,
    scaler_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    aa_to_idx = {aa: idx for idx, aa in enumerate(AA)}
    sum_x, sum_x2, count = init_stats(aa_continuous_values.shape[1])

    for record in tqdm(records, total=len(records)):
        aa_idx = sequence_to_indices(record.sequence, aa_to_idx)

        if aa_binary_values.shape[1] > 0:
            save_array(aa_bin_dir / f"{record.stem}.npy", aa_binary_values[aa_idx].astype(np.float32, copy=False))

        if aa_continuous_values.shape[1] > 0:
            raw = aa_continuous_values[aa_idx].astype(np.float32, copy=False)
            save_array(aa_cont_dir / f"{record.stem}.npy", raw)
            update_stats(sum_x, sum_x2, count, raw)

    means, stds = finish_stats(sum_x, sum_x2, count, 1e-8)
    dump({"feature_names": aa_continuous_names, "mean": means, "std": stds}, scaler_path)

    return means, stds


def encode_interaction_matrices(
    records: list[Record],
    matrix_values: np.ndarray,
    matrix_feature_names: list[str],
    matrix_dir: Path,
    scaler_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    aa_to_idx = {aa: idx for idx, aa in enumerate(MATRIX_AA)}
    sum_x, sum_x2, count = init_stats(matrix_values.shape[1])

    for record in tqdm(records, total=len(records)):
        aa_idx = sequence_to_indices(record.sequence, aa_to_idx)
        raw = matrix_values[aa_idx].astype(np.float32, copy=False)
        save_array(matrix_dir / f"{record.stem}.npy", raw)
        update_stats(sum_x, sum_x2, count, raw)

    means, stds = finish_stats(sum_x, sum_x2, count, 1.0)
    dump({"feature_names": matrix_feature_names, "mean": means, "std": stds}, scaler_path)

    return means, stds


def write_final_npz_files(
    records: list[Record],
    output_dir: Path,
    paths: dict[str, Path],
    aa_binary_names: list[str],
    aa_continuous_names: list[str],
    matrix_feature_names: list[str],
    aa_means: np.ndarray,
    aa_stds: np.ndarray,
    matrix_means: np.ndarray,
    matrix_stds: np.ndarray,
) -> None:
    feature_names = np.array(
        ["rel_pos", "dist_to_term", "sinusoidal_pos"]
        + [f"bin_{aa}" for aa in AA]
        + aa_binary_names
        + aa_continuous_names
        + matrix_feature_names,
        dtype=np.str_,
    )

    print("outputting npz files of all samples...")
    for record in tqdm(records, total=len(records)):
        parts = [
            load_array(paths["pos"] / f"{record.stem}.npy").astype(np.float32, copy=False),
            load_array(paths["bins"] / f"{record.stem}.npy").astype(np.float32, copy=False),
        ]

        if aa_binary_names:
            parts.append(load_array(paths["aa_bin"] / f"{record.stem}.npy").astype(np.float32, copy=False))

        if aa_continuous_names:
            aa_cont = load_array(paths["aa_cont"] / f"{record.stem}.npy").astype(np.float32, copy=False)
            parts.append(((aa_cont - aa_means) / aa_stds).astype(np.float32, copy=False))

        if matrix_feature_names:
            matrix = load_array(paths["matrix"] / f"{record.stem}.npy").astype(np.float32, copy=False)
            parts.append(((matrix - matrix_means) / matrix_stds).astype(np.float32, copy=False))

        X = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        np.savez_compressed(
            output_dir / f"{record.stem}.npz",
            X=X,
            y=record.target,
            feature_names=feature_names,
        )

    shutil.rmtree(paths["tmp_root"], ignore_errors=True)


def main() -> None:
    args = parse_args()

    records = load_records(args.input_csv)
    lookup_tables = load_lookup_tables(args.indices_txt, args.lookup_tables_txt)
    aa_binary_values, aa_continuous_values, aa_binary_names, aa_continuous_names = classify_lookup_tables(lookup_tables)
    matrix_values, matrix_feature_names = load_matrices(args.matrix_dir)
    paths = prepare_dirs(args.output_dir, args.scaler_dir)

    aa_scaler_path = args.scaler_dir / "aa_indices_continuous_scaler.joblib"
    matrix_scaler_path = args.scaler_dir / "interaction_matrices_scaler.joblib"

    print("encoding amino acid bins...")
    encode_amino_acid_bins(records, paths["bins"])

    print("encoding positional features...")
    encode_positional_features(records, paths["pos"])

    print("encoding amino acid indices...")
    aa_means, aa_stds = encode_amino_acid_indices(
        records,
        aa_binary_values,
        aa_continuous_values,
        aa_continuous_names,
        paths["aa_bin"],
        paths["aa_cont"],
        aa_scaler_path,
    )

    print("encoding interaction matrices...")
    matrix_means, matrix_stds = encode_interaction_matrices(
        records,
        matrix_values,
        matrix_feature_names,
        paths["matrix"],
        matrix_scaler_path,
    )

    write_final_npz_files(
        records,
        args.output_dir,
        paths,
        aa_binary_names,
        aa_continuous_names,
        matrix_feature_names,
        aa_means,
        aa_stds,
        matrix_means,
        matrix_stds,
    )


if __name__ == "__main__":
    main()
