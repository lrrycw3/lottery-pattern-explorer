#!/usr/bin/env python3
"""Nightly lottery results updater (runs in GitHub Actions).

Reads the official Michigan Lottery Past Results tool with Playwright
and merges any missing draws into data/fantasy5_history.csv and
data/daily5_history.csv.

Source (official Michigan Lottery only, never a third-party site):
the Past Results tool, queried separately for the three games:
  - Fantasy 5:        SELECTED_GAME=5 (Regular + Double Play rows)
  - Daily 5 Midday:   SELECTED_GAME=J
  - Daily 5 Evening:  SELECTED_GAME=2
The tool is interactive (no deep-linkable results URL), so the script
fills the Start/End date fields, clicks "Get Results", and reads the
official "Download" CSV export for the requested range. The export is
machine-readable, so no numbers are ever scraped from ball graphics.

Range: from each dataset's latest date through today
(America/Detroit). Multi-day gaps self-repair from the range query.

Format preserved per file: UTF-8 BOM, all fields quoted, newest
first, Fantasy 5 Regular row before Double Play per date, Daily 5
Midday before Evening per date. Daily 5 digits stay in drawn order
and are never sorted or deduplicated; Fantasy 5 numbers are stored
exactly as the official export lists them.

Exits 0 with no changes when everything is already current. Exits
non-zero (failing the Action run) when the source cannot be read or
a downloaded row fails validation. It never invents numbers: a row
is only added if it was read from michiganlottery.com in this run.
"""

import csv
import sys
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

from playwright.sync_api import sync_playwright

BASE_URL = (
    "https://www.michiganlottery.com/resources/number-tools"
    "?SELECTED_TOOL=PAST_RESULTS&SELECTED_GAME={code}"
)
REPO_ROOT = Path(__file__).resolve().parent.parent
DETROIT = ZoneInfo("America/Detroit")

# game code -> (csv path, drawing types in file order, validator)
GAMES = {
    "5": ("data/fantasy5_history.csv",
          ["REGULAR DRAWING", "DOUBLE PLAY DRAWING"], "fantasy5"),
    "J": ("data/daily5_history.csv",
          ["MIDDAY DRAWING"], "daily5"),
    "2": ("data/daily5_history.csv",
          ["EVENING DRAWING"], "daily5"),
}

# Full drawing-type order per file, for newest-first sorting. (Daily 5
# is one file shared by two game codes, so the sort order cannot come
# from a single game's type list.)
FILE_TYPE_ORDER = {
    "data/fantasy5_history.csv": ["REGULAR DRAWING", "DOUBLE PLAY DRAWING"],
    "data/daily5_history.csv": ["MIDDAY DRAWING", "EVENING DRAWING"],
}


def today_detroit() -> date:
    from datetime import datetime
    return datetime.now(DETROIT).date()


def parse_date(s: str) -> date:
    month, day, year = (int(x) for x in s.strip().split("/"))
    return date(year, month, day)


def fmt_date(d: date) -> str:
    return d.strftime("%m/%d/%Y")


def read_csv_rows(path: Path):
    """Existing rows as (date, numbers_str, type) plus file line ending."""
    raw = path.read_bytes()
    eol = b"\r\n" if raw.count(b"\r\n") > 0 else b"\n"
    text = raw.decode("utf-8-sig")
    rows = []
    for row in csv.DictReader(StringIO(text)):
        rows.append((parse_date(row["Draw Date"]),
                     row["Winning Numbers"].strip(),
                     row["Drawing Type"].strip()))
    return rows, eol


def write_csv_rows(path: Path, rows, eol: bytes):
    """rows: (date, numbers_str, type), newest first. BOM + all quoted."""
    out = StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL,
                        lineterminator=eol.decode())
    writer.writerow(["Draw Date", "Winning Numbers", "Drawing Type"])
    for d, nums, dtype in rows:
        writer.writerow([fmt_date(d), nums, dtype])
    path.write_bytes(b"\xef\xbb\xbf" + out.getvalue().encode("utf-8"))


def validate_fantasy5(numbers: str):
    parts = numbers.split(",")
    if len(parts) != 5:
        raise ValueError(f"Fantasy 5 needs 5 numbers: {numbers!r}")
    nums = [int(p) for p in parts]
    if len(set(nums)) != 5 or any(n < 1 or n > 39 for n in nums):
        raise ValueError(f"Fantasy 5 numbers out of range: {numbers!r}")
    return ",".join(str(n) for n in nums)


def validate_daily5(numbers: str):
    parts = numbers.split(",")
    if len(parts) != 5:
        raise ValueError(f"Daily 5 needs 5 digits: {numbers!r}")
    digits = [int(p) for p in parts]
    if any(d < 0 or d > 9 for d in digits):
        raise ValueError(f"Daily 5 digits out of range: {numbers!r}")
    return ",".join(str(d) for d in digits)


def fetch_game(page, code: str, start: date, end: date):
    """Download the official Past Results CSV for one game + range.

    Returns [(date, numbers_str, drawing_type_or_None)] in file order.
    """
    page.goto(BASE_URL.format(code=code), wait_until="domcontentloaded")
    get_results = page.get_by_role("button", name="Get Results")
    get_results.wait_for(timeout=60000)
    page.get_by_role("textbox", name="Start Date").fill(fmt_date(start))
    page.get_by_role("textbox", name="End Date").fill(fmt_date(end))
    # Blur the fields so the date picker parses the typed values.
    page.get_by_role("textbox", name="End Date").press("Tab")
    get_results.wait_for(state="visible")
    if not get_results.is_enabled():
        raise RuntimeError(
            f"Get Results stayed disabled for game {code} "
            f"({fmt_date(start)} to {fmt_date(end)})")
    with page.expect_download(timeout=60000) as dl_info:
        get_results.click()
        page.get_by_role("link", name="Download").first.wait_for(
            timeout=60000)
        page.get_by_role("link", name="Download").first.click()
    download = dl_info.value
    tmp = REPO_ROOT / f".past-results-{code}.csv"
    download.save_as(tmp)
    text = tmp.read_bytes().decode("utf-8-sig")
    tmp.unlink()
    rows = []
    for row in csv.DictReader(StringIO(text)):
        d = parse_date(row["Draw Date"])
        nums = row["Winning Numbers"].strip()
        dtype = (row.get("Drawing Type") or "").strip() or None
        rows.append((d, nums, dtype))
    if not rows:
        raise RuntimeError(
            f"No rows downloaded for game {code} "
            f"({fmt_date(start)} to {fmt_date(end)})")
    return rows


def merge_game(code: str, path: Path, types, kind: str,
               downloaded, start: date, end: date) -> int:
    """Merge downloaded rows into the dataset file. Returns rows added."""
    rel = str(path.relative_to(REPO_ROOT))
    order = FILE_TYPE_ORDER[rel]
    existing, eol = read_csv_rows(path)
    have = {(d, t) for d, _, t in existing}
    latest = max(d for d, _, _ in existing)
    if start < latest:
        start = latest  # only ever fill forward from the latest we hold
    added = []
    validate = (validate_fantasy5 if kind == "fantasy5"
                else validate_daily5)
    if kind == "fantasy5":
        # The export may or may not carry a Drawing Type column. With
        # it, use it; without it, rows come in displayed order per
        # date: Regular first, then Double Play.
        by_date = {}
        for d, nums, dtype in downloaded:
            by_date.setdefault(d, []).append((nums, dtype))
        for d in sorted(by_date):
            if d < start or d > end:
                continue
            entries = by_date[d]
            for i, (nums, dtype) in enumerate(entries):
                if dtype is None:
                    if len(entries) != 2:
                        raise RuntimeError(
                            f"Fantasy 5 {fmt_date(d)}: expected 2 rows "
                            f"without a type column, got {len(entries)}")
                    dtype = types[i]
                if dtype not in types:
                    raise RuntimeError(
                        f"Unexpected Fantasy 5 type {dtype!r}")
                if (d, dtype) in have:
                    continue
                added.append((d, validate(nums), dtype))
    else:
        dtype = types[0]
        for d, nums, _ in downloaded:
            if d < start or d > end:
                continue
            if (d, dtype) in have:
                continue
            added.append((d, validate(nums), dtype))
    if not added:
        return 0
    # Newest date first; within a date, file order (Regular before
    # Double Play, Midday before Evening). The negated index plus
    # reverse=True achieves both at once.
    merged = sorted(existing + added,
                    key=lambda r: (r[0], -order.index(r[2])),
                    reverse=True)
    write_csv_rows(path, merged, eol)
    return len(added)


def main() -> int:
    today = today_detroit()
    # Per-file latest dates, so each game queries only what it lacks.
    latest_by_file = {}
    for code, (rel, _, _) in GAMES.items():
        path = REPO_ROOT / rel
        rows, _ = read_csv_rows(path)
        latest_by_file[rel] = max(d for d, _, _ in rows)
    total_added = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/126.0 Safari/537.36"))
        try:
            for code, (rel, types, kind) in GAMES.items():
                path = REPO_ROOT / rel
                start = latest_by_file[rel]
                if start >= today:
                    print(f"game {code}: {rel} already current "
                          f"through {fmt_date(start)}")
                    continue
                downloaded = fetch_game(page, code, start, today)
                added = merge_game(code, path, types, kind,
                                   downloaded, start, today)
                total_added += added
                print(f"game {code}: added {added} row(s) to {rel}")
        finally:
            browser.close()
    print(f"total rows added: {total_added}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
