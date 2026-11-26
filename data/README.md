# IMDB Movie Reviews Dataset

## Source
IMDB Dataset of 50K Movie Reviews (`IMDB Dataset.csv`, Apna College).
Place the downloaded file inside `data/raw/IMDB Dataset.csv` (ignored by Git).

## Schema
| Column | Type | Description |
| :--- | :--- | :--- |
| `review` | string | Raw movie review text (contains HTML tags such as `<br />`) |
| `sentiment` | string | Binary target label (`positive` or `negative`) |

## EDA Summary (`notebooks/01_eda.ipynb`)
- **Missing values:** 0
- **Unique rows (after dropping 418 duplicates):** 49,582
- **Class balance:** 24,884 positive / 24,698 negative
- **Word count stats:** median = 173, mean = 231, 75th percentile = 281, max = 2,470