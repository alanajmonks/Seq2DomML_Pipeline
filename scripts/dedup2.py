from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from Bio.Align import PairwiseAligner
from tqdm import tqdm


INPUT_CSV = Path("data/preprocessing/targ_encoded.csv")
OUTPUT_CSV = Path("data/preprocessing/targ_deduped.csv")

SEQ_COL = "Sequence"

IDENTITY_THRESHOLD = 0.90
MAX_LEN_DIFF = 10
PREFIX_LEN = 50
PREFIX_ID_THR = 0.75
CHUNK_SIZE = 20_000
MAX_WORKERS = 12

_WORKER_SEQS = None
_WORKER_PREFIX_LEN = None
_WORKER_PREFIX_THR = None
_WORKER_FINAL_THR = None


def prefix_identity(a: str, b: str, n: int) -> float:
    if not a or not b:
        return 0.0
    a = a[:n]
    b = b[:n]
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / length


def make_candidate_pairs_by_length(lengths: list[int], max_len_diff: int) -> list[tuple[int, int]]:
    buckets = defaultdict(list)
    for idx, length in enumerate(lengths):
        buckets[length].append(idx)

    keys = sorted(buckets)
    pairs: list[tuple[int, int]] = []

    for i, length in enumerate(keys):
        current = buckets[length]

        if len(current) > 1:
            for pos, left_idx in enumerate(current):
                for right_idx in current[pos + 1 :]:
                    pairs.append((left_idx, right_idx))

        j = i + 1
        while j < len(keys) and keys[j] - length <= max_len_diff:
            other = buckets[keys[j]]
            for left_idx in current:
                for right_idx in other:
                    pairs.append((left_idx, right_idx))
            j += 1

    return pairs


def iter_chunks(items: list[tuple[int, int]], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def init_worker(seqs: list[str], prefix_len: int, prefix_thr: float, final_thr: float) -> None:
    global _WORKER_SEQS, _WORKER_PREFIX_LEN, _WORKER_PREFIX_THR, _WORKER_FINAL_THR
    _WORKER_SEQS = seqs
    _WORKER_PREFIX_LEN = prefix_len
    _WORKER_PREFIX_THR = prefix_thr
    _WORKER_FINAL_THR = final_thr


def worker_chunk(chunk: list[tuple[int, int]]) -> list[tuple[int, int]]:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = 0.0
    aligner.extend_gap_score = 0.0

    passed: list[tuple[int, int]] = []
    for i, j in chunk:
        s1 = _WORKER_SEQS[i]
        s2 = _WORKER_SEQS[j]

        if prefix_identity(s1, s2, _WORKER_PREFIX_LEN) < _WORKER_PREFIX_THR:
            continue

        score = aligner.score(s1, s2)
        identity = score / max(len(s1), len(s2))
        if identity >= _WORKER_FINAL_THR:
            passed.append((i, j))

    return passed


def find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent: list[int], rank: list[int], a: int, b: int) -> None:
    ra = find(parent, a)
    rb = find(parent, b)

    if ra == rb:
        return

    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(1)

    df = pd.read_csv(INPUT_CSV)
    if SEQ_COL not in df.columns:
        raise SystemExit(1)

    df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()
    df = df[df[SEQ_COL].str.len() > 0].reset_index(drop=True)

    before_count = len(df)
    print(f"before deduplicating: {before_count}")

    if before_count == 0:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print("after deduplicating: 0")
        return

    seqs = df[SEQ_COL].tolist()
    lengths = [len(seq) for seq in seqs]
    candidate_pairs = make_candidate_pairs_by_length(lengths, MAX_LEN_DIFF)

    parent = list(range(before_count))
    rank = [0] * before_count

    if candidate_pairs:
        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=init_worker,
            initargs=(seqs, PREFIX_LEN, PREFIX_ID_THR, IDENTITY_THRESHOLD),
        ) as executor:
            future_sizes = {}
            for chunk in iter_chunks(candidate_pairs, CHUNK_SIZE):
                future = executor.submit(worker_chunk, chunk)
                future_sizes[future] = len(chunk)

            with tqdm(total=len(candidate_pairs)) as pbar:
                for future in as_completed(future_sizes):
                    for i, j in future.result():
                        union(parent, rank, i, j)
                    pbar.update(future_sizes[future])

    keep_idx = []
    seen_roots = set()
    for i in range(before_count):
        root = find(parent, i)
        if root not in seen_roots:
            seen_roots.add(root)
            keep_idx.append(i)

    df_out = df.iloc[keep_idx].reset_index(drop=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"after deduplicating: {len(df_out)}")


if __name__ == "__main__":
    main()
