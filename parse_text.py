"""Parser for the ``catalog_content`` text blob in the price-prediction CSV.

Each row's ``catalog_content`` is a newline-separated pseudo-record, e.g.::

    Item Name: Goya Foods Sazonador Total Seasoning, 30 Ounce (Pack of 6)
    Bullet Point 1: ...
    Bullet Point 2: ...
    Value: 180.0
    Unit: Ounce

Fields can be missing (no ``Value:`` line, ``Unit: None``, ``Value: nan``,
zero bullet points, or — rarely — an ``Item Name N:`` variant instead of a
plain ``Item Name:``). :func:`parse_catalog_content` handles all of these
gracefully and always returns a fully-populated dict.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# `Item Name:` in the overwhelming majority of rows; a handful of rows instead
# use `Item Name 1:` / `Item Name 2:` (e.g. an internal SKU vs. a display
# name). We match both and, when several are present, keep the longest
# captured string since that is reliably the more descriptive name.
_ITEM_NAME_RE = re.compile(r"Item Name(?:\s*\d+)?:\s*(.+?)(?:\n|$)")
_VALUE_RE = re.compile(r"Value:\s*([0-9]*\.?[0-9]+)")
_UNIT_RE = re.compile(r"Unit:\s*(.+?)(?:\n|$)")
_BULLET_RE = re.compile(r"Bullet Point\s*\d+:\s*(.+?)(?:\n|$)")
_PACK_OF_RE = re.compile(r"Pack of\s*(\d+)", re.IGNORECASE)

_MISSING_UNIT_TOKENS = {"", "none", "nan", "null"}


def parse_catalog_content(text: Any) -> dict[str, Any]:
    """Parse one ``catalog_content`` string into structured fields.

    Returns a dict with keys:
        item_name (str): best-effort display name, "" if absent.
        value (Optional[float]): parsed quantity value, None if absent/"nan".
        unit (str): normalized-but-cased unit string, "" if absent/"None".
        details (str): bullet points joined by a single space, "" if none.
        n_bullets (int): number of bullet points found.
    """
    if not isinstance(text, str) or not text.strip():
        return {"item_name": "", "value": None, "unit": "", "details": "", "n_bullets": 0}

    item_matches = _ITEM_NAME_RE.findall(text)
    item_name = max((m.strip() for m in item_matches), key=len, default="")

    value_match = _VALUE_RE.search(text)
    value: Optional[float] = float(value_match.group(1)) if value_match else None

    unit_match = _UNIT_RE.search(text)
    unit = unit_match.group(1).strip() if unit_match else ""
    if unit.lower() in _MISSING_UNIT_TOKENS:
        unit = ""

    bullets = [b.strip() for b in _BULLET_RE.findall(text)]
    bullets = [b for b in bullets if b]
    details = " ".join(bullets)

    return {
        "item_name": item_name,
        "value": value,
        "unit": unit,
        "details": details,
        "n_bullets": len(bullets),
    }


def extract_pack_count(item_name: Any) -> int:
    """Extract a pack count from an item name via `Pack of (\\d+)`; default 1."""
    if not isinstance(item_name, str):
        return 1
    m = _PACK_OF_RE.search(item_name)
    if not m:
        return 1
    n = int(m.group(1))
    if n <= 0:
        return 1
    return min(n, 100)  # defensive cap against rare text-parsing outliers


if __name__ == "__main__":
    samples = [
        "Item Name: Goya Foods Sazonador Total Seasoning, 30 Ounce (Pack of 6)\n"
        "Bullet Point 1: Great for cooking\n"
        "Bullet Point 2: Family recipe\n"
        "Value: 180.0\n"
        "Unit: Ounce\n",
        "Item Name: Seasons Brand Imported Skinless & Boneless Sardines in Water No Salt Added -- 4.25 oz - 3PC\n"
        "Value: nan\n"
        "Unit: None\n",
        "Item Name 1: Item# 704-4295\nItem Name 2: Icing Pouch with Tips, 8 oz.\n"
        "Bullet Point 1: 1 pouch ready-to-use icing\n"
        "Value: 1.0\nUnit: Count\n",
        "",
        None,
    ]

    r0 = parse_catalog_content(samples[0])
    assert r0["item_name"] == "Goya Foods Sazonador Total Seasoning, 30 Ounce (Pack of 6)"
    assert r0["value"] == 180.0
    assert r0["unit"] == "Ounce"
    assert r0["n_bullets"] == 2
    assert extract_pack_count(r0["item_name"]) == 6

    r1 = parse_catalog_content(samples[1])
    assert r1["value"] is None
    assert r1["unit"] == ""

    r2 = parse_catalog_content(samples[2])
    assert r2["item_name"] == "Icing Pouch with Tips, 8 oz."  # longest match wins
    assert extract_pack_count(r2["item_name"]) == 1

    r3 = parse_catalog_content(samples[3])
    assert r3 == {"item_name": "", "value": None, "unit": "", "details": "", "n_bullets": 0}

    r4 = parse_catalog_content(samples[4])
    assert r4["item_name"] == ""

    for i, r in enumerate([r0, r1, r2, r3, r4]):
        print(f"sample {i}: {r}")
    print("All parse_text self-tests passed.")
