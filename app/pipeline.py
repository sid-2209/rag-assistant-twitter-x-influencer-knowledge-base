from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


RAW_DIR_DEFAULT = Path("data/raw")
PROCESSED_PATH_DEFAULT = Path("data/processed/processed.json")


@dataclass
class RawRecord:
    id: Optional[str]
    name: str
    handle: str
    followers: Optional[int]
    niche: Optional[str]
    sample_post: Optional[str]


def _ensure_at_prefix(handle: str) -> str:
    handle = handle.strip()
    if not handle:
        return handle
    return handle if handle.startswith("@") else f"@{handle}"


def _clean_whitespace(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return " ".join(str(text).split())


def _normalize_niche(niche: Any) -> Optional[str]:
    if niche is None:
        return None
    if isinstance(niche, list):
        items = [str(x).strip().lower() for x in niche if str(x).strip()]
        return ", ".join(items) if items else None
    # assume string
    items = [seg.strip().lower() for seg in str(niche).split(",")]
    items = [i for i in items if i]
    return ", ".join(items) if items else None


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except Exception:
        return None


def _chunk_text(text: Optional[str], max_len: int) -> List[str]:
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for token in text.split():
        token_len = len(token) + (1 if length > 0 else 0)
        if length + token_len > max_len:
            chunks.append(" ".join(current))
            current = [token]
            length = len(token)
        else:
            current.append(token)
            length += token_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def _load_one_file(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend([dict(x) for x in data])
            elif isinstance(data, dict):
                for key in ("data", "records", "items"):
                    if isinstance(data.get(key), list):
                        records.extend([dict(x) for x in data[key]])
                        break
        except Exception:
            return []
    elif path.suffix.lower() == ".csv":
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(dict(row))
        except Exception:
            return []
    return records


def load_raw_records(raw_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not raw_dir.exists():
        return records
    for path in sorted(raw_dir.glob("**/*")):
        records.extend(_load_one_file(path))
    return records


def load_raw_records_from_path(path: Path) -> List[Dict[str, Any]]:
    if path.is_dir():
        return load_raw_records(path)
    if path.is_file():
        return _load_one_file(path)
    return []


def normalize_and_validate(records: Iterable[Dict[str, Any]], max_chunk_len: int = 280) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen_handles: Set[str] = set()
    seen_name_keys: Set[str] = set()

    for rec in records:
        raw = RawRecord(
            id=str(rec.get("id")) if rec.get("id") is not None else None,
            name=_clean_whitespace(rec.get("name") or "") or "",
            handle=_clean_whitespace(rec.get("handle") or "") or "",
            followers=_parse_int(rec.get("followers")),
            niche=_normalize_niche(rec.get("niche")),
            sample_post=_clean_whitespace(rec.get("sample_post")) or _clean_whitespace(rec.get("post")),
        )

        # Required fields
        if not raw.name or not raw.handle:
            continue

        handle_norm = _ensure_at_prefix(raw.handle.lower())
        name_key = raw.name.strip().lower()

        # Dedup by handle primarily, fallback to name
        if handle_norm in seen_handles or (handle_norm == "@" and name_key in seen_name_keys):
            continue

        seen_handles.add(handle_norm)
        seen_name_keys.add(name_key)

        chunks = _chunk_text(raw.sample_post, max_chunk_len)

        normalized.append(
            {
                "id": raw.id,
                "name": raw.name,
                "handle": handle_norm,
                "followers": raw.followers,
                "niche": raw.niche,
                "sample_post": raw.sample_post,
                "chunks": chunks,
            }
        )

    return normalized


def write_processed(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def run_pipeline(input_path: Path, output_file: Path, max_chunk_len: int) -> Tuple[int, int]:
    raw = load_raw_records_from_path(input_path)
    processed = normalize_and_validate(raw, max_chunk_len=max_chunk_len)
    write_processed(processed, output_file)
    return (len(raw), len(processed))


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL pipeline: raw → processed")
    parser.add_argument("--input", type=Path, default=RAW_DIR_DEFAULT, help="Input directory or file (JSON/CSV)")
    parser.add_argument("--input-dir", type=Path, default=None, help="Deprecated alias for --input (directory only)")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=PROCESSED_PATH_DEFAULT,
        help="Path to write processed JSON file",
    )
    parser.add_argument(
        "--max-chunk-len",
        type=int,
        default=280,
        help="Max characters per chunk for long sample_post text",
    )
    args = parser.parse_args()

    effective_input: Path
    if args.input_dir is not None:
        effective_input = args.input_dir
    else:
        effective_input = args.input

    total_raw, total_processed = run_pipeline(effective_input, args.output_file, args.max_chunk_len)
    print(f"Processed {total_processed} of {total_raw} raw records → {args.output_file}")


if __name__ == "__main__":
    main()
