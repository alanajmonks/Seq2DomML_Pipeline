import csv
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm

GRAPHQL_URL = "https://data.rcsb.org/graphql"
INPUT_DIR = Path("data/pdb_entry_ids")
OUTPUT_DIR = Path("data/preprocessing")
OUTPUT_CSV = OUTPUT_DIR / "sequences.csv"
OUTPUT_LOG = OUTPUT_DIR / "preprocessing_log.txt"
BATCH_SIZE = 250
VALID_INPUT_SUFFIXES = {".txt", ".csv"}
ENTRY_ID_PATTERN = re.compile(r"^[A-Z0-9]{4}$")
EXCLUSION_PATTERN = re.compile(r"\b(?:membrane|transmembrane|lipoprotein|lipoproteins|lipo[-\s]?protein(?:s)?|lipo)\b", re.IGNORECASE)
QUERY = """
query($entry_ids:[String!]!){
  entries(entry_ids:$entry_ids){
    rcsb_id
    struct{
      title
    }
    struct_keywords{
      pdbx_keywords
    }
    polymer_entities{
      entity_poly{
        pdbx_seq_one_letter_code_can
        rcsb_entity_polymer_type
      }
      rcsb_polymer_entity{
        pdbx_description
        details
      }
      rcsb_polymer_entity_container_identifiers{
        auth_asym_ids
      }
    }
  }
}
"""

def batched(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]

def get_input_files():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")
    files = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in VALID_INPUT_SUFFIXES
    )
    if not files:
        raise ValueError(f"No supported input files found in {INPUT_DIR}")
    return files

def read_input_text(path):
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode input file: {path}")

def load_entry_ids():
    entry_ids = []
    seen = set()
    files = get_input_files()
    for path in tqdm(files, desc="Reading ID files", unit="file"):
        text = read_input_text(path)
        file_count = 0
        for line in text.splitlines():
            for token in re.split(r"[\s,;]+", line.strip().upper()):
                if ENTRY_ID_PATTERN.fullmatch(token) and token not in seen:
                    seen.add(token)
                    entry_ids.append(token)
                    file_count += 1
        tqdm.write(f"Loaded {file_count} IDs from {path.name}")
    if not entry_ids:
        raise ValueError(f"No PDB entry IDs found in {INPUT_DIR}")
    return entry_ids

def fetch_entries(batch):
    payload = json.dumps({"query": QUERY, "variables": {"entry_ids": batch}}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "pdb-preprocess/1.0",
    }
    last_error = None
    for attempt in range(5):
        try:
            request = Request(GRAPHQL_URL, data=payload, headers=headers)
            with urlopen(request, timeout=60) as response:
                result = json.load(response)
            errors = result.get("errors") or []
            if errors:
                raise RuntimeError("; ".join(error.get("message", "GraphQL error") for error in errors))
            return result.get("data", {}).get("entries") or []
        except Exception as exc:
            last_error = exc
            if attempt == 4:
                raise last_error
            tqdm.write(f"Retrying batch starting at {batch[0]} after error: {exc}")
            time.sleep(2 ** attempt)
    raise last_error

def is_protein(entity):
    polymer_type = ((entity.get("entity_poly") or {}).get("rcsb_entity_polymer_type") or "").lower()
    return "protein" in polymer_type or "polypeptide" in polymer_type

def get_sequence(entity):
    sequence = ((entity.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can") or "")
    return re.sub(r"\s+", "", sequence)

def get_chain_ids(entity):
    chain_ids = ((entity.get("rcsb_polymer_entity_container_identifiers") or {}).get("auth_asym_ids") or [])
    if isinstance(chain_ids, str):
        chain_ids = [chain_ids]
    return list(dict.fromkeys(chain_ids))

def should_exclude(entry, entity):
    text = " ".join(
        value
        for value in [
            ((entry.get("struct") or {}).get("title") or ""),
            ((entry.get("struct_keywords") or {}).get("pdbx_keywords") or ""),
            ((entity.get("rcsb_polymer_entity") or {}).get("pdbx_description") or ""),
            ((entity.get("rcsb_polymer_entity") or {}).get("details") or ""),
        ]
        if value
    )
    return bool(EXCLUSION_PATTERN.search(text))

def main():
    tqdm.write("Loading entry IDs")
    entry_ids = load_entry_ids()
    tqdm.write(f"Loaded {len(entry_ids)} unique entry IDs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_records = set()
    total_chains_found = 0
    chains_removed = 0

    tqdm.write("Fetching entries from RCSB and collecting all chains")
    with tqdm(total=len(entry_ids), desc="Processing entries", unit="entry") as progress:
        for batch in batched(entry_ids, BATCH_SIZE):
            for entry in fetch_entries(batch):
                if not entry:
                    continue
                entry_id = (entry.get("rcsb_id") or "").upper()
                for entity in entry.get("polymer_entities") or []:
                    if not is_protein(entity):
                        continue
                    sequence = get_sequence(entity)
                    if not sequence:
                        continue
                    excluded = should_exclude(entry, entity)
                    for chain_id in get_chain_ids(entity):
                        if not chain_id:
                            continue
                        total_chains_found += 1
                        if excluded:
                            chains_removed += 1
                            continue
                        record_id = f"{entry_id}_{chain_id}"
                        if record_id in seen_records:
                            continue
                        seen_records.add(record_id)
                        rows.append({"Entry ID": record_id, "Sequence": sequence})
            progress.update(len(batch))
            progress.set_postfix(chains=total_chains_found, kept=len(rows), removed=chains_removed)

    tqdm.write(f"Writing CSV to {OUTPUT_CSV}")
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Entry ID", "Sequence"])
        writer.writeheader()
        writer.writerows(rows)

    tqdm.write(f"Writing log to {OUTPUT_LOG}")
    OUTPUT_LOG.write_text(
        "\n".join(
            [
                f"entry_ids_inputted: {len(entry_ids)}",
                f"total_chains_found: {total_chains_found}",
                f"chains_removed_lipo_or_membrane: {chains_removed}",
                f"chains_written: {len(rows)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    tqdm.write("Finished preprocessing")

if __name__ == "__main__":
    main()
