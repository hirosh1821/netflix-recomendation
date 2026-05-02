import argparse
from pathlib import Path

import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    factors = np.load(args.artifacts_dir / "svd_factors.npz")
    mappings = joblib.load(args.artifacts_dir / "id_mappings.joblib")
    users = list(map(str, mappings["users"]))
    movies = list(map(str, mappings["movies"]))

    if str(args.user_id) not in users:
        raise ValueError(f"Unknown user_id: {args.user_id}")

    user_idx = users.index(str(args.user_id))
    scores = factors["movie_factors"] @ factors["user_factors"][user_idx]
    scores = scores + float(factors["global_mean"][0])
    top_indices = np.argsort(scores)[::-1][: args.top_k]

    for rank, movie_idx in enumerate(top_indices, start=1):
        print(f"{rank:02d}. movie_id={movies[movie_idx]} predicted_rating={scores[movie_idx]:.3f}")


if __name__ == "__main__":
    main()
