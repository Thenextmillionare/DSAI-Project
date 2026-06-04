# IE0005 Toxic Comment Multi-Label Classification

This project classifies online comments into multiple toxicity categories:

- toxic
- severe toxic
- obscene
- threat
- insult
- identity hate

The notebook covers exploratory data analysis, text cleaning, TF-IDF feature
engineering, Naive Bayes and Logistic Regression models, threshold tuning,
bias and fairness evaluation, and class imbalance optimisation.

## Files

- `IE0005-toxic-comment-classification.ipynb` - final project notebook
- `requirements.txt` - Python dependencies

## Dataset

Place the following Jigsaw Toxic Comment Classification Challenge files in the
repository folder before running the notebook:

- `train.csv`
- `test.csv`
- `test_labels.csv`

These dataset files are intentionally excluded from Git because they are large.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open `IE0005-toxic-comment-classification.ipynb` and run the cells in order.
