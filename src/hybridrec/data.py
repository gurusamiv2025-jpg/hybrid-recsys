"""Download and load MovieLens 1M, plus (optionally) the ML-25M tag genome.

MovieLens movie ids are stable across GroupLens releases, so the tag-genome
relevance vectors shipped with ML-25M can be joined onto ML-1M items by id.
"""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


@dataclass
class MovieLens:
    ratings: pd.DataFrame  # user_id, item_id, rating, ts
    movies: pd.DataFrame   # item_id, title, year, genres (list[str])
    users: pd.DataFrame    # user_id, gender, age, occupation
    genome: pd.DataFrame | None = None  # index item_id, columns = genome tags


def _fetch_range(url: str, start: int, end: int, retries: int = 5) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
        except OSError:
            if attempt == retries - 1:
                raise
    raise IOError(f"incomplete range {start}-{end} from {url}")


def _download(url: str, dest: Path, workers: int = 16, chunk: int = 4 << 20) -> Path:
    """Download to ``dest`` (skipped if present). Large files are fetched as
    parallel HTTP range requests, because files.grouplens.org throttles each
    connection; the result is written to a .part file and renamed when complete."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        size = int(r.headers.get("Content-Length", 0))
        ranged = r.headers.get("Accept-Ranges") == "bytes"
    print(f"Downloading {url} ({size / 1e6:.0f} MB) -> {dest}")
    if not ranged or size < 2 * chunk:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while block := r.read(1 << 20):
                f.write(block)
    else:
        spans = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]
        with open(tmp, "wb") as f, ThreadPoolExecutor(workers) as pool:
            f.truncate(size)
            futures = {pool.submit(_fetch_range, url, s, e): s for s, e in spans}
            for n, fut in enumerate(as_completed(futures), 1):
                f.seek(futures[fut])
                f.write(fut.result())
                if n % 10 == 0 or n == len(spans):
                    print(f"  {n}/{len(spans)} chunks")
    tmp.rename(dest)
    return dest


def _read_dat(z: zipfile.ZipFile, name: str, columns: list[str]) -> pd.DataFrame:
    # The .dat files use a multi-char "::" separator; swapping it for a tab lets
    # the fast C parser handle them.
    text = z.read(name).decode("latin-1").replace("::", "\t")
    return pd.read_csv(io.StringIO(text), sep="\t", header=None, names=columns,
                       quoting=3)


def load_ml1m(data_dir: str | Path) -> MovieLens:
    data_dir = Path(data_dir)
    path = _download(ML1M_URL, data_dir / "ml-1m.zip")
    with zipfile.ZipFile(path) as z:
        ratings = _read_dat(z, "ml-1m/ratings.dat", ["user_id", "item_id", "rating", "ts"])
        movies = _read_dat(z, "ml-1m/movies.dat", ["item_id", "title", "genres"])
        users = _read_dat(z, "ml-1m/users.dat", ["user_id", "gender", "age", "occupation", "zip"])

    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    movies["year"] = year.fillna(year.median()).astype(int)
    movies["title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    movies["genres"] = movies["genres"].str.split("|")
    users = users.drop(columns="zip")
    return MovieLens(ratings=ratings, movies=movies, users=users)


def load_genome(data_dir: str | Path, item_ids: np.ndarray) -> pd.DataFrame:
    """Tag-genome relevance matrix (items x 1128 tags) for the given ML movie ids.

    The 260 MB ML-25M archive is downloaded once; the slice we need is cached
    as a small compressed .npz so later runs skip the big CSV.
    """
    data_dir = Path(data_dir)
    cache = data_dir / "genome_ml1m.npz"
    if not cache.exists():
        path = _download(ML25M_URL, data_dir / "ml-25m.zip")
        print("Extracting tag genome from ML-25M (one-off, ~30s)...")
        with zipfile.ZipFile(path) as z:
            tags = pd.read_csv(z.open("ml-25m/genome-tags.csv"))
            scores = pd.read_csv(z.open("ml-25m/genome-scores.csv"),
                                 dtype={"movieId": np.int32, "tagId": np.int16,
                                        "relevance": np.float32})
        scores = scores[scores["movieId"].isin(set(item_ids.tolist()))]
        mat = scores.pivot(index="movieId", columns="tagId", values="relevance")
        np.savez_compressed(cache, item_ids=mat.index.to_numpy(),
                            tag_ids=mat.columns.to_numpy(),
                            tags=tags.set_index("tagId").loc[mat.columns, "tag"].to_numpy(),
                            values=mat.to_numpy(np.float32))
    npz = np.load(cache, allow_pickle=True)
    return pd.DataFrame(npz["values"], index=npz["item_ids"], columns=npz["tags"])


def load(data_dir: str | Path, use_genome: bool = True) -> MovieLens:
    ml = load_ml1m(data_dir)
    if use_genome:
        ml.genome = load_genome(data_dir, ml.movies["item_id"].to_numpy())
    return ml


def tokenize_title(title: str) -> str:
    """Lower-cased title with leading/trailing articles normalised ("Matrix, The")."""
    title = re.sub(r",\s*(The|A|An|Les|La|Le|Il|Das|Der|Die|El)$", "", title)
    return title.lower()
