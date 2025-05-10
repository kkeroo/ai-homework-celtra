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