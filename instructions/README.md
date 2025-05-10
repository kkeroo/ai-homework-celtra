# Overview

We use OpenAI embedder and a Logistic Regression classifier to classify company names into predefined categories. To get the features we use Wikipedia API to obtain the summary of the company and use the first 3 sentences of the description as features. The 3 sentences are forwarded to OpenAI embedder to get the embeddings. Embeddings are used as features for the classifier.

Acchieved accuracy is 0.56.

## Installation

You need a `OPENAI_API_KEY` env. variable set or the `api_key` variable in `source/main.py` needs to be set.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the model use the following command:

```bash
python source/main.py
```

## Results

| Category | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Aerospace & defense | 0.00 | 0.00 | 0.00 | 4 |
| Banking | 0.50 | 0.94 | 0.66 | 63 |
| Business services & supplies | 0.40 | 0.14 | 0.21 | 14 |
| Capital goods | 0.38 | 0.27 | 0.32 | 11 |
| Chemicals | 0.67 | 0.20 | 0.31 | 10 |
| Conglomerates | 0.00 | 0.00 | 0.00 | 6 |
| Construction | 0.64 | 0.44 | 0.52 | 16 |
| Consumer durables | 0.50 | 0.53 | 0.52 | 15 |
| Diversified financials | 0.33 | 0.55 | 0.41 | 31 |
| Drugs & biotechnology | 0.83 | 0.56 | 0.67 | 9 |
| Food drink & tobacco | 0.62 | 0.88 | 0.73 | 17 |
| Food markets | 1.00 | 0.14 | 0.25 | 7 |
| Health care equipment & services | 0.67 | 0.77 | 0.71 | 13 |
| Hotels restaurants & leisure | 1.00 | 0.43 | 0.60 | 7 |
| Household & personal products | 1.00 | 0.22 | 0.36 | 9 |
| Insurance | 0.70 | 0.73 | 0.71 | 22 |
| Materials | 0.55 | 0.58 | 0.56 | 19 |
| Media | 0.67 | 0.33 | 0.44 | 12 |
| Oil & gas operations | 0.75 | 0.67 | 0.71 | 18 |
| Retailing | 0.43 | 0.33 | 0.38 | 18 |
| Semiconductors | 1.00 | 0.20 | 0.33 | 5 |
| Software & services | 0.00 | 0.00 | 0.00 | 6 |
| Technology hardware & equipment | 0.20 | 0.08 | 0.12 | 12 |
| Telecommunications services | 0.60 | 0.69 | 0.64 | 13 |
| Trading companies | 0.00 | 0.00 | 0.00 | 5 |
| Transportation | 0.63 | 0.75 | 0.69 | 16 |
| Utilities | 0.84 | 0.73 | 0.78 | 22 |
| **Accuracy** | | | **0.56** | **400** |
| **Macro avg** | **0.55** | **0.41** | **0.43** | **400** |
| **Weighted avg** | **0.56** | **0.56** | **0.52** | **400** |