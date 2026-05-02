import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def load_ratings(path: Path) -> pd.DataFrame:
    ratings = pd.read_csv(path)
    required = {"user_id", "movie_id", "rating"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return ratings[list(required)].dropna()


def encode_ids(ratings: pd.DataFrame):
    users = pd.Categorical(ratings["user_id"])
    movies = pd.Categorical(ratings["movie_id"])
    encoded = ratings.assign(user_idx=users.codes, movie_idx=movies.codes)
    return encoded, list(users.categories), list(movies.categories)


def build_matrix(ratings: pd.DataFrame, n_users: int, n_movies: int) -> csr_matrix:
    return csr_matrix(
        (ratings["rating"].astype(float), (ratings["user_idx"], ratings["movie_idx"])),
        shape=(n_users, n_movies),
    )


def predict_pairs(user_factors, movie_factors, rows: pd.DataFrame, global_mean: float) -> np.ndarray:
    preds = np.sum(user_factors[rows["user_idx"].to_numpy()] * movie_factors[rows["movie_idx"].to_numpy()], axis=1)
    return np.clip(preds + global_mean, 1.0, 5.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--components", type=int, default=80)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    ratings = load_ratings(args.ratings)
    encoded, users, movies = encode_ids(ratings)
    train, test = train_test_split(encoded, test_size=args.test_size, random_state=42, stratify=None)

    global_mean = float(train["rating"].mean())
    train_centered = train.assign(rating=train["rating"] - global_mean)
    matrix = build_matrix(train_centered, len(users), len(movies))

    svd = TruncatedSVD(n_components=args.components, random_state=42)
    user_factors = svd.fit_transform(matrix)
    movie_factors = svd.components_.T

    preds = predict_pairs(user_factors, movie_factors, test, global_mean)
    y_true = test["rating"].astype(float).to_numpy()
    metrics = {
        "rows": int(len(ratings)),
        "users": int(len(users)),
        "movies": int(len(movies)),
        "components": int(args.components),
        "rmse": float(mean_squared_error(y_true, preds, squared=False)),
        "mae": float(mean_absolute_error(y_true, preds)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
