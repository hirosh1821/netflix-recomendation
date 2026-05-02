# Netflix Recommendation System

Matrix-factorisation recommendation pipeline for the Netflix Prize dataset.

## What this project does

- Loads Netflix-style `user_id`, `movie_id`, `rating` interactions.
- Builds sparse train/test splits.
- Trains an SVD collaborative-filtering model.
- Evaluates with RMSE and MAE.
- Generates top-N movie recommendations for a user.

## Repository structure

```text
src/
  train_recommender.py
reports/
  metrics.json
notebooks/
```

## Dataset

This project expects a ratings CSV with these columns:

```text
user_id,movie_id,rating
```

The original Netflix Prize dataset is large and is not committed to this repo.
Place the dataset at:

```text
data/ratings.csv
```

## Quick start

```bash
pip install -r requirements.txt
python src/train_recommender.py --ratings data/ratings.csv --output reports/metrics.json
python src/recommend.py --user-id u1 --top-k 5
```

## Resume-ready summary

Built an end-to-end collaborative-filtering recommendation system using SVD-style matrix factorisation, sparse interaction modelling, and RMSE/MAE evaluation for personalised movie ranking.
