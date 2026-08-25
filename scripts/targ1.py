import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


DATA_DIR = Path("data")
SEQ_FILE = DATA_DIR / "preprocessing/sequences.csv"
DOM_FILE = DATA_DIR / "target_labeling/ids_from_ecod.txt"
OUT_CSV = DATA_DIR / "preprocessing/targ_encoded.csv"

MAX_WORKERS = 12
BOUNDARY_LEN = 9
MIN_LINKER_LEN = 7
REPEATED_KEY_POLICY = "last"

ID_COL = "Entry ID"
SEQ_COL = "Sequence"
LABEL_COL = "targ_binary"


def canonical_key(raw: str) -> Optional[str]:
    txt = str(raw).strip().upper()
    if not txt:
        return None

    m = re.match(r"^([A-Z0-9]{4})[_:](.+)$", txt)
    if m:
        pdb, chain = m.groups()
        chain = chain.strip().upper()
        return f"{pdb}_{chain}" if chain else None

    m = re.match(r"^([A-Z0-9]{4})(.+)$", txt)
    if m:
        pdb, chain = m.groups()
        chain = chain.strip().upper()
        return f"{pdb}_{chain}" if chain else None

    return None


def normalize_chain_token(raw: str) -> str:
    txt = str(raw).strip().upper()
    txt = txt.replace("CHAIN", "")
    txt = re.sub(r"[^A-Z0-9]", "", txt)
    return txt


def chains_match_exact(expected: str, observed: str) -> bool:
    return bool(expected) and bool(observed) and expected == observed


def parse_sequences_csv(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path)
    if ID_COL not in df.columns or SEQ_COL not in df.columns:
        raise ValueError("sequences.csv must contain 'Entry ID' and 'Sequence' columns")

    seqs: Dict[str, str] = {}
    for _, row in df.iterrows():
        raw_id = str(row[ID_COL]).strip()
        seq = str(row[SEQ_COL]).strip().upper()
        key = canonical_key(raw_id) or raw_id.strip().upper()

        if not seq:
            continue

        if key in seqs and seqs[key] != seq:
            continue

        seqs[key] = seq

    return seqs


def parse_ids_with_domains(
    path: Path,
    repeated_key_policy: str,
) -> Dict[str, List[Tuple[int, int]]]:
    if repeated_key_policy not in {"first", "last", "merge"}:
        raise ValueError("repeated_key_policy must be one of: first, last, merge")

    dom_map: Dict[str, List[Tuple[int, int]]] = {}
    range_pat = re.compile(r"([^\s:;,]+)\s*:\s*(-?\d+)\s*-\s*(-?\d+)")

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or ";" not in line:
                continue

            left, right = line.split(";", 1)
            left_key = canonical_key(left)
            if not left_key:
                continue

            expected_chain = normalize_chain_token(left_key.split("_", 1)[1])
            tokens = range_pat.findall(right)
            if not tokens:
                continue

            matched_ranges: List[Tuple[int, int]] = []
            for rhs_chain_raw, a_raw, b_raw in tokens:
                rhs_chain = normalize_chain_token(rhs_chain_raw)
                if not chains_match_exact(expected_chain, rhs_chain):
                    continue

                a = int(a_raw)
                b = int(b_raw)
                s, e = (a, b) if a <= b else (b, a)
                matched_ranges.append((s, e))

            if not matched_ranges:
                continue

            matched_ranges.sort(key=lambda x: (x[0], x[1]))

            if left_key not in dom_map:
                dom_map[left_key] = matched_ranges
            elif repeated_key_policy == "last":
                dom_map[left_key] = matched_ranges
            elif repeated_key_policy == "merge":
                combined = dom_map[left_key] + matched_ranges
                dom_map[left_key] = sorted(set(combined), key=lambda x: (x[0], x[1]))

    return dom_map


def normalize_intervals(
    intervals: Sequence[Tuple[int, int]],
    seq_len: int,
) -> List[Tuple[int, int]]:
    cleaned: List[Tuple[int, int]] = []

    for s, e in intervals:
        lo, hi = (s, e) if s <= e else (e, s)

        if hi < 1 or lo > seq_len:
            continue

        lo = max(1, lo)
        hi = min(seq_len, hi)

        if lo <= hi:
            cleaned.append((lo, hi))

    if not cleaned:
        return []

    cleaned.sort()
    merged: List[Tuple[int, int]] = [cleaned[0]]
    for s, e in cleaned[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    return merged


def domain_to_core_intervals(
    domains: Sequence[Tuple[int, int]],
    boundary_len: int,
) -> List[List[int]]:
    cores: List[List[int]] = []
    for s, e in domains:
        cs = s + boundary_len
        ce = e - boundary_len
        if cs <= ce:
            cores.append([cs, ce])
    return cores


def enforce_min_linkers(
    cores: List[List[int]],
    min_linker_len: int,
) -> None:
    if min_linker_len <= 0 or len(cores) < 2:
        return

    for i in range(len(cores) - 1):
        left = cores[i]
        right = cores[i + 1]

        gap = right[0] - left[1] - 1
        if gap >= min_linker_len:
            continue

        need = min_linker_len - gap
        left_cap = max(0, left[1] - left[0] + 1)
        right_cap = max(0, right[1] - right[0] + 1)

        if left_cap + right_cap == 0:
            continue

        target_left = need // 2
        target_right = need - target_left

        take_left = min(left_cap, target_left)
        take_right = min(right_cap, target_right)
        remaining = need - (take_left + take_right)

        if remaining > 0:
            extra_left = min(left_cap - take_left, remaining)
            take_left += extra_left
            remaining -= extra_left

        if remaining > 0:
            extra_right = min(right_cap - take_right, remaining)
            take_right += extra_right

        if take_left > 0:
            left[1] -= take_left

        if take_right > 0:
            right[0] += take_right


def label_sequence_binary(
    seq: str,
    raw_domains: Sequence[Tuple[int, int]],
    boundary_len: int,
    min_linker_len: int,
) -> np.ndarray:
    n = len(seq)
    labels = np.ones(n, dtype=np.int8)

    merged_domains = normalize_intervals(raw_domains, n)
    cores = domain_to_core_intervals(merged_domains, boundary_len)
    enforce_min_linkers(cores, min_linker_len)

    for s, e in cores:
        if s <= e:
            labels[s - 1 : e] = 0

    return labels


def process_entry(
    key: str,
    seqs: Dict[str, str],
    doms: Dict[str, List[Tuple[int, int]]],
    boundary_len: int,
    min_linker_len: int,
) -> Tuple[str, Optional[dict]]:
    seq = seqs.get(key)
    raw_domains = doms.get(key)

    if not seq or not raw_domains:
        return key, None

    y = label_sequence_binary(
        seq=seq,
        raw_domains=raw_domains,
        boundary_len=boundary_len,
        min_linker_len=min_linker_len,
    )

    return key, {
        ID_COL: key,
        SEQ_COL: seq,
        LABEL_COL: "".join(map(str, y.tolist())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-file", type=Path, default=SEQ_FILE)
    parser.add_argument("--dom-file", type=Path, default=DOM_FILE)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--boundary-len", type=int, default=BOUNDARY_LEN)
    parser.add_argument("--min-linker-len", type=int, default=MIN_LINKER_LEN)
    parser.add_argument(
        "--repeated-key-policy",
        type=str,
        default=REPEATED_KEY_POLICY,
        choices=["first", "last", "merge"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.seq_file.exists() or not args.dom_file.exists():
        raise SystemExit("Missing sequences.csv or ids_with_domains.txt")

    print("encoding target variable across sequences...")

    seqs = parse_sequences_csv(args.seq_file)
    doms = parse_ids_with_domains(args.dom_file, args.repeated_key_policy)
    keys = list(seqs.keys())
    row_map: Dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [
            ex.submit(
                process_entry,
                key,
                seqs,
                doms,
                args.boundary_len,
                args.min_linker_len,
            )
            for key in keys
        ]

        with tqdm(total=len(futures)) as pbar:
            for fut in as_completed(futures):
                key, row = fut.result()
                if row is not None:
                    row_map[key] = row
                pbar.update(1)

    out_rows = [row_map[key] for key in keys if key in row_map]
    if not out_rows:
        raise SystemExit("No entries produced.")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows, columns=[ID_COL, SEQ_COL, LABEL_COL]).to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
