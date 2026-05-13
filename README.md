# Comparative Sentiment Analysis — Product Reviews

A systematic benchmark comparing **Classical ML**, **Deep Learning**, and **Transformer** models on the `amazon_polarity` dataset.

## Dataset
- **Source:** `amazon_polarity` (HuggingFace)
- **Task:** Binary sentiment classification (Positive / Negative)
- **Working subset:** 20 000 train · 5 000 test (stratified, seed=42)

## Models Compared

| Tier | Models |
|---|---|
| Classical ML | Logistic Regression, SVM (linear), Multinomial Naive Bayes |
| Deep Learning | BiLSTM, TextCNN |
| Transformers | DistilBERT, BERT-base-uncased, RoBERTa-base |

## Project Structure

```
copestone/
├── data/
│   └── prepare_data.py          # Download & subsample amazon_polarity
├── models/
│   ├── classical/
│   │   └── train_classical.py   # TF-IDF + Logistic Regression / SVM / NB
│   ├── deep_learning/
│   │   ├── dataset.py           # PyTorch Dataset + GloVe loader
│   │   ├── bilstm.py            # BiLSTM model definition
│   │   ├── textcnn.py           # TextCNN model definition
│   │   └── train_dl.py          # Training loop
│   └── transformers/
│       └── train_transformer.py # HuggingFace fine-tuning
├── evaluate/
│   └── evaluate_all.py          # Unified metrics & comparison plots
├── results/                     # Saved metrics JSON + plots (auto-generated)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download & prepare the dataset
python data/prepare_data.py

# 3. Train all model tiers
python models/classical/train_classical.py
python models/deep_learning/train_dl.py --model bilstm
python models/deep_learning/train_dl.py --model textcnn
python models/transformers/train_transformer.py --model distilbert-base-uncased

# 4. Compare results
python evaluate/evaluate_all.py
```

## Evaluation Metrics
- Accuracy · Macro F1 · ROC-AUC · Inference latency (ms/sample)
