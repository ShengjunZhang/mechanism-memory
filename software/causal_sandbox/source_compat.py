"""Check recorded source identities after documented module renaming."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _manifest():
    path = ROOT / "release_manifest.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _record(name):
    name = str(name).replace("\\", "/")
    path = Path(name)
    if path.is_absolute():
        try:
            name = path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return None
    records = _manifest().get("source_transforms", {})
    matches = [(old, r) for old, r in records.items()
               if name in (old, r["path"]) or
               ("/" not in name and name in (Path(old).name, Path(r["path"]).name))]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous source name: {name}")
    return matches[0] if matches else None


def _restore(content, record):
    lines = content.decode("utf-8").splitlines(keepends=True)
    for edit in reversed(record["reverse_edits"]):
        lines[edit["start"]:edit["end"]] = edit["original"]
    return "".join(lines).encode("utf-8")


def recorded_source_matches(path, expected):
    record = _record(path)
    if record:
        _, info = record
        target = ROOT / info["path"]
    else:
        target = Path(str(path).replace("\\", "/"))
        if not target.is_absolute():
            target = ROOT / target
    if not target.is_file():
        return False
    content = target.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual == expected:
        return True
    return bool(record and expected == info["original_sha256"]
                and actual == info["packaged_sha256"]
                and hashlib.sha256(_restore(content, info)).hexdigest() == expected)


def _normalized(hashes):
    normalized = {}
    for name, checksum in hashes.items():
        key = str(name).replace("\\", "/")
        match = _record(key)
        if match:
            _, record = match
            if checksum not in (record["original_sha256"], record["packaged_sha256"]):
                return None
            if not recorded_source_matches(key, checksum):
                return None
            key = record["path"] if "/" in key else Path(record["path"]).name
            checksum = record["packaged_sha256"]
        if key in normalized:
            return None
        normalized[key] = checksum
    return normalized


def sources_match(first, second):
    a, b = _normalized(first), _normalized(second)
    return a is not None and b is not None and a == b


def records_match(first, second):
    if first.keys() != second.keys():
        return False
    return all(sources_match(first[key], second[key]) if key in ("source_hashes", "code_hashes")
               else first[key] == second[key] for key in first)


def verify_source_transforms():
    records = _manifest().get("source_transforms", {})
    for name, record in records.items():
        assert recorded_source_matches(name, record["original_sha256"]), name
    return len(records)
