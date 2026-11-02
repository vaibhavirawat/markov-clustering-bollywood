"""Scrape Hindi film casts from Wikipedia 'List of Hindi films of YEAR' pages.

Writes data/films.csv with columns: year, title, cast (actors joined by '|').
Raw wikitext is cached in data/raw/ so re-runs are offline.
"""
import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "SNA-course-mini-project/1.0 (https://github.com/vaibhavirawat/markov-clustering-bollywood)"}
YEARS = range(2000, 2025)

LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")
REF = re.compile(r"<ref[^>/]*?/>|<ref.*?</ref>", re.S)
COMMENT = re.compile(r"<!--.*?-->", re.S)


def fetch(year):
    cache = RAW / f"{year}.txt"
    if cache.exists():
        return cache.read_text()
    url = "https://en.wikipedia.org/w/api.php"
    params = dict(action="parse", page=f"List_of_Hindi_films_of_{year}",
                  prop="wikitext", format="json", formatversion=2)
    r = requests.get(url, params=params, headers=HEADERS, timeout=40)
    r.raise_for_status()
    data = r.json()
    if "parse" not in data:
        return ""
    text = data["parse"]["wikitext"]
    cache.write_text(text)
    time.sleep(0.5)
    return text


def split_cells(row):
    """Split one wikitext table row into cells (handles '||' and newline '|')."""
    cells = []
    for line in row.split("\n"):
        line = line.strip()
        if not line.startswith("|") or line.startswith("|}") or line.startswith("|+"):
            continue
        cells.extend(c.strip() for c in line[1:].split("||"))
    return cells


def parse_year(text, year):
    text = COMMENT.sub("", REF.sub("", text))
    films = []
    for row in re.split(r"\n\|-+[^\n]*", text):
        cells = split_cells(row)
        # Title cell is the italic wikilink: ''[[Title]]''
        idx = next((i for i, c in enumerate(cells) if re.search(r"''\s*\[\[", c) or re.match(r"\[\[''", c)), None)
        if idx is None or idx + 2 >= len(cells):
            continue
        title_m = LINK.search(cells[idx])
        if not title_m:
            continue
        title = (title_m.group(2) or title_m.group(1)).replace("''", "").strip()
        cast = []
        for m in LINK.finditer(cells[idx + 2]):
            name = m.group(1).strip()
            if ":" in name:
                continue
            cast.append(re.sub(r"\s*\((actor|actress|.*?)\)$", "", name))
        if len(cast) >= 2:
            films.append((year, title, "|".join(dict.fromkeys(cast))))
    return films


def main():
    rows = []
    for y in YEARS:
        try:
            text = fetch(y)
        except Exception as e:  # network hiccup: skip year, report it
            print(f"{y}: FAILED ({e})", file=sys.stderr)
            continue
        films = parse_year(text, y)
        print(f"{y}: {len(films)} films with >=2 linked cast")
        rows.extend(films)
    with open(HERE / "data" / "films.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "title", "cast"])
        w.writerows(rows)
    print("total films:", len(rows))


if __name__ == "__main__":
    main()
