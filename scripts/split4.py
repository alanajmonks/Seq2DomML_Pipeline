from pathlib import Path
import shutil

import numpy as np
from tqdm import tqdm


INPUT_DIR = Path("data/full_dataset_with_features")
OUTPUT_ROOT = Path("data/split_dataset")
TMP_OUTPUT_ROOT = OUTPUT_ROOT.with_name(f"{OUTPUT_ROOT.name}__tmp")

TRAIN_DIR = OUTPUT_ROOT / "train"
TEST_AND_VAL_DIR = OUTPUT_ROOT / "test_and_val"
FEAT_NAMES_FILE = OUTPUT_ROOT / "feat_names"
TRAIN_IDS_FILE = OUTPUT_ROOT / "train_entry_IDs"
TEST_AND_VAL_IDS_FILE = OUTPUT_ROOT / "test_and_val_entry_IDs"

TMP_TRAIN_DIR = TMP_OUTPUT_ROOT / "train"
TMP_TEST_AND_VAL_DIR = TMP_OUTPUT_ROOT / "test_and_val"
TMP_FEAT_NAMES_FILE = TMP_OUTPUT_ROOT / "feat_names"
TMP_TRAIN_IDS_FILE = TMP_OUTPUT_ROOT / "train_entry_IDs"
TMP_TEST_AND_VAL_IDS_FILE = TMP_OUTPUT_ROOT / "test_and_val_entry_IDs"

TEST_AND_VAL_PROP = 0.35
RNG_SEED = 1337


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_npz_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files found in {path}")
    return files


def split_entry_ids(unique_ids: np.ndarray, rng: np.random.Generator) -> tuple[set[str], set[str]]:
    total_ids = len(unique_ids)
    n_test_and_val_ids = int(round(TEST_AND_VAL_PROP * total_ids))
    n_test_and_val_ids = max(1, min(n_test_and_val_ids, total_ids - 1))

    test_and_val_ids = set(rng.choice(unique_ids, size=n_test_and_val_ids, replace=False).tolist())
    train_ids = set(unique_ids.tolist()) - test_and_val_ids

    if train_ids & test_and_val_ids:
        raise RuntimeError("Train and test_and_val overlap")
    if len(train_ids) + len(test_and_val_ids) != total_ids:
        raise RuntimeError("Split sizes do not sum to total input size")

    return train_ids, test_and_val_ids


def write_id_manifest(path: Path, ids: set[str]) -> None:
    path.write_text("".join(f"{entry_id}\n" for entry_id in sorted(ids)), encoding="utf-8")


def normalize_feature_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def write_feature_names(npz_path: Path, out_path: Path) -> None:
    with np.load(npz_path, allow_pickle=True) as z:
        if "feature_names" in z.files:
            raw_names = z["feature_names"].reshape(-1).tolist()
        elif "feat_names" in z.files:
            raw_names = z["feat_names"].reshape(-1).tolist()
        else:
            raise RuntimeError(f"No feature name field found in {npz_path}")

    lines = [f"{idx}\t{normalize_feature_name(name)}" for idx, name in enumerate(raw_names)]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def verify_dir_contents(path: Path, expected_ids: set[str]) -> None:
    actual_ids = {file.stem for file in path.glob("*.npz")}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:10]
        extra = sorted(actual_ids - expected_ids)[:10]
        raise RuntimeError(
            f"Directory contents mismatch for {path}. "
            f"missing examples={missing} extra examples={extra}"
        )


def replace_output_path(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    src.replace(dst)


def build_tmp_output_tree() -> None:
    reset_dir(TMP_OUTPUT_ROOT)
    TMP_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TMP_TEST_AND_VAL_DIR.mkdir(parents=True, exist_ok=True)


def commit_tmp_output_tree() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    replace_output_path(TMP_TRAIN_DIR, TRAIN_DIR)
    replace_output_path(TMP_TEST_AND_VAL_DIR, TEST_AND_VAL_DIR)
    replace_output_path(TMP_FEAT_NAMES_FILE, FEAT_NAMES_FILE)
    replace_output_path(TMP_TRAIN_IDS_FILE, TRAIN_IDS_FILE)
    replace_output_path(TMP_TEST_AND_VAL_IDS_FILE, TEST_AND_VAL_IDS_FILE)

    if TMP_OUTPUT_ROOT.exists():
        shutil.rmtree(TMP_OUTPUT_ROOT)


def main() -> None:
    files = list_npz_files(INPUT_DIR)

    unique_ids = np.array([file.stem for file in files], dtype=object)
    rng = np.random.default_rng(RNG_SEED)
    train_ids, test_and_val_ids = split_entry_ids(unique_ids, rng)

    build_tmp_output_tree()
    write_feature_names(files[0], TMP_FEAT_NAMES_FILE)

    train_count = 0
    test_and_val_count = 0

    with tqdm(total=len(files), desc="Copying split dataset") as pbar:
        for file in files:
            entry_id = file.stem

            if entry_id in train_ids:
                shutil.copy2(file, TMP_TRAIN_DIR / file.name)
                train_count += 1
            elif entry_id in test_and_val_ids:
                shutil.copy2(file, TMP_TEST_AND_VAL_DIR / file.name)
                test_and_val_count += 1
            else:
                raise RuntimeError(f"{entry_id} not assigned to either split")

            pbar.update(1)

    if train_count != len(train_ids):
        raise RuntimeError(
            f"Train count mismatch: expected {len(train_ids)}, copied {train_count}"
        )
    if test_and_val_count != len(test_and_val_ids):
        raise RuntimeError(
            f"test_and_val count mismatch: expected {len(test_and_val_ids)}, copied {test_and_val_count}"
        )
    if train_count + test_and_val_count != len(files):
        raise RuntimeError(
            f"Total output mismatch: input {len(files)}, output {train_count + test_and_val_count}"
        )

    verify_dir_contents(TMP_TRAIN_DIR, train_ids)
    verify_dir_contents(TMP_TEST_AND_VAL_DIR, test_and_val_ids)

    write_id_manifest(TMP_TRAIN_IDS_FILE, train_ids)
    write_id_manifest(TMP_TEST_AND_VAL_IDS_FILE, test_and_val_ids)

    commit_tmp_output_tree()

    print(f"train: {train_count}")
    print(f"test_and_val: {test_and_val_count}")


if __name__ == "__main__":
    main()
