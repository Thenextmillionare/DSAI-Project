# IE0005 Toxic Comment Multi-Label Classification

An end-to-end machine learning project for detecting toxic Wikipedia comments
across six overlapping toxicity categories.

This project was completed by **DSAI EL17** as part of the IE0005 mini project.
It explores how traditional natural language processing techniques can support
content moderation while also examining the limitations and fairness risks of
automated toxicity detection.

## Problem Statement

Online comments can contain several forms of harmful language at the same time.
For example, one comment may be both toxic, obscene, and insulting. This makes
the task a **multi-label classification** problem rather than a standard
single-label classification problem.

The model predicts the following labels:

| Label | Description |
| --- | --- |
| `toxic` | General toxic or harmful language |
| `severe_toxic` | Highly aggressive or severely toxic language |
| `obscene` | Obscene or profane language |
| `threat` | Threatening language |
| `insult` | Insulting or abusive language |
| `identity_hate` | Hate directed at an identity group |

## Dataset

The project uses the
[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
dataset, which contains comments from Wikipedia discussion pages.

| File | Rows | Purpose |
| --- | ---: | --- |
| `train.csv` | 159,571 | Labelled comments used for training and validation |
| `test.csv` | 153,164 | Unlabelled comments used for final predictions |
| `test_labels.csv` | 153,164 | Ground-truth test labels, including unevaluated rows marked `-1` |

After removing unevaluated test rows, **63,978 comments** are available for
final test-set evaluation.

The dataset files are not committed to this repository because they are large
and are distributed separately by Kaggle.

## Project Workflow

The notebook implements a complete machine learning pipeline:

1. **Data loading and validation**
   - Loads the training, test, and test-label datasets.
   - Inspects shapes, columns, missing values, and summary statistics.

2. **Exploratory data analysis**
   - Compares toxic and non-toxic comment frequencies.
   - Examines the distribution of each toxicity label.
   - Measures label correlations to show why a multi-label approach is needed.
   - Identifies severe class imbalance, especially for `threat` and
     `identity_hate`.

3. **Text cleaning**
   - Converts text to lowercase.
   - Removes HTML tags, numbers, punctuation, and extra whitespace.
   - Expands common contractions so that negation is better preserved.
   - Removes empty training comments after cleaning.

4. **Feature engineering**
   - Extracts character count and word count.
   - Counts exclamation marks and question marks.
   - Calculates uppercase-character ratio as a possible aggression signal.

5. **Text vectorisation**
   - Uses TF-IDF with up to 30,000 unigram and bigram features.
   - Preserves important negation words such as `no`, `not`, `nor`, and
     `never`.
   - Combines sparse TF-IDF features with scaled numerical features.

6. **Model development**
   - Trains Multinomial Naive Bayes as a baseline model.
   - Trains one-vs-rest Logistic Regression as the main model.
   - Uses class weighting to improve sensitivity to minority labels.

7. **Evaluation and optimisation**
   - Evaluates precision, recall, F1-score, and ROC-AUC for each label.
   - Tunes per-label decision thresholds.
   - Uses a stratified split, random oversampling, and Logistic Regression
     hyperparameter tuning.
   - Inspects model weights for interpretability.
   - Evaluates false-positive rates for identity-related terms.

## Model Results

Logistic Regression outperformed the Naive Bayes baseline for every toxicity
label on the validation set.

| Metric | Naive Bayes | Logistic Regression |
| --- | ---: | ---: |
| Macro F1 | 0.4754 | 0.4830 |
| Mean ROC-AUC | 0.9567 | 0.9761 |

### Validation ROC-AUC by Label

| Label | Naive Bayes | Logistic Regression |
| --- | ---: | ---: |
| `toxic` | 0.9506 | 0.9643 |
| `severe_toxic` | 0.9746 | 0.9852 |
| `obscene` | 0.9594 | 0.9841 |
| `threat` | 0.9497 | 0.9787 |
| `insult` | 0.9549 | 0.9721 |
| `identity_hate` | 0.9509 | 0.9725 |

On the labelled test set, Logistic Regression achieved ROC-AUC values between
**0.9499 and 0.9859** across the six labels. The model intentionally favours
recall over precision because missing genuinely toxic content can be more
harmful than sending additional comments for review.

## Key Findings

- The dataset is heavily imbalanced: most comments are non-toxic, while labels
  such as `threat` and `identity_hate` have relatively few training examples.
- Logistic Regression benefits from combining TF-IDF text features with simple
  writing-style features.
- Bigrams help capture some context, but the model still struggles with
  negation and sentence meaning because it is fundamentally a bag-of-words
  approach.
- Learned feature weights show that profanity strongly drives toxic
  predictions, while common Wikipedia editorial language is associated with
  non-toxic comments.
- Per-label threshold tuning improves the usefulness of predictions beyond a
  single default threshold of `0.5`.

## Bias and Fairness

Automated toxicity detection can incorrectly associate identity terms with
harmful language. The notebook measures false-positive rates for several
identity-related terms and finds that some groups are flagged much more often
than the overall baseline.

Notable findings include:

- Overall false-positive-rate baseline: **0.1227**
- Comments containing `gay`: **0.5696** false-positive rate
- Comments containing `woman`: **0.3349** false-positive rate
- Comments containing `man`: **0.1276** false-positive rate

These results show that strong overall model performance does not guarantee
fair behaviour across different groups. Threshold adjustment can reduce some
effects, but it does not remove bias learned from the training data.

## Limitations

- TF-IDF does not fully understand context, word order, sarcasm, or negation.
- Rare labels have limited examples and are harder to classify reliably.
- Class weighting and oversampling may increase false positives.
- Identity-related terms can become unfairly associated with toxicity.
- The model should support human moderation rather than make fully autonomous
  enforcement decisions.

## Future Improvements

- Fine-tune a context-aware transformer model such as BERT or DistilBERT.
- Apply adversarial debiasing or counterfactual data augmentation.
- Collect more representative examples for rare toxicity categories.
- Compare oversampling methods such as SMOTE with class weighting.
- Build an ensemble that combines complementary model predictions.
- Add an interactive interface for reviewing comment predictions and model
  confidence.

## Repository Contents

| File | Description |
| --- | --- |
| `IE0005-toxic-comment-classification.ipynb` | Final notebook with code, explanations, charts, and saved results |
| `requirements.txt` | Python dependencies required to run the notebook |
| `.gitignore` | Excludes virtual environments, notebook checkpoints, and dataset CSV files |

## Setup and Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Thenextmillionare/DSAI-Project.git
   cd DSAI-Project
   ```

2. Create and activate a virtual environment.

   Windows:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   macOS or Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download `train.csv`, `test.csv`, and `test_labels.csv` from Kaggle and
   place them in the repository root.

5. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Open `IE0005-toxic-comment-classification.ipynb` and run the cells in order.

## Team Contributions

| Team Member | Contribution |
| --- | --- |
| Sriman | Dataset selection, exploratory data analysis, visualisation, cleaning, feature engineering, and TF-IDF |
| Aravinth | Naive Bayes baseline, Logistic Regression, threshold pipeline, and bias evaluation |
| Lavantika | Precision, recall, F1-score, ROC-AUC evaluation, threshold analysis, and model comparison |
| Louise | Optimisation pipeline, stratified splitting, oversampling, and hyperparameter tuning |
