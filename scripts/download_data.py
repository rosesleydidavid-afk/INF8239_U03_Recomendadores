from __future__ import annotations

import json
from datetime import datetime, timezone

from inf8239_u03.config import MOVIELENS_DIR, MOVIELENS_URL, ROOT
from inf8239_u03.data import download_zip, load_movielens

destination, digest = download_zip(MOVIELENS_URL, ROOT / "data/raw")
ratings, movies = load_movielens(MOVIELENS_DIR)
provenance = {
    "url": MOVIELENS_URL,
    "downloaded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    "sha256_of_zip": digest,
    "ratings_rows": int(len(ratings)),
    "movies_rows": int(len(movies)),
    "users": int(ratings["userId"].nunique()),
}
reports = ROOT / "reports"
reports.mkdir(exist_ok=True)
(reports / "dataset_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
print("Destino:", destination)
print("SHA-256:", digest)
print(json.dumps(provenance, indent=2))
