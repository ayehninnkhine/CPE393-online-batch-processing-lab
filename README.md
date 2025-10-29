
# ML Model Serving: Online vs Batch Prediction

## What it does
- Trains a Logistic Regression model to predict `passed` from `hours` studied.
- **Online prediction:** predict a single value (simulating per-request inference) and print **per-call latency**.
- **Batch prediction:** predict a CSV of inputs and write all predictions to `predictions.csv`, reporting **total time** and **throughput**.

## Setup
```bash
pip install scikit-learn pandas joblib
```

## 1) Train the model
```bash
python train_model.py
```
Outputs: `model.joblib`

## 2) Online prediction (one value at a time)
```bash
python predict_online.py --hours 5
# Try different hours: 2, 4, 6, 8 ...
```

## 3) Batch prediction (many rows)
```bash
# Use the provided sample CSV
python predict_batch.py --input data/batch_inputs.csv --output predictions.csv
```

After running batch, open `predictions.csv` to see inputs and predicted labels.


### **Goal:** Understand difference between **online** and **batch** machine learning prediction modes.

Run:

```bash
python train_model.py
```

### Q1:

What accuracy does the model show on the test set?

Run:

```bash
python predict_online.py --hours 5
python predict_online.py --hours 2
python predict_online.py --hours 8
```

### Q2:

Record your results:

| Hours | Probability (pass_prob) | Prediction (0/1) | Latency (ms) |
| ----- | ----------------------- | ---------------- | ------------ |
| 5     |                         |                  |              |
| 2     |                         |                  |              |
| 8     |                         |                  |              |


Run:

```bash
python predict_batch.py --input batch_inputs.csv --output predictions.csv
```

### Q3:

How many rows were processed?

### Q4:

What was the **total compute time** and **predictions per second**?


### Q5:
Imagine a **fraud detection system** that must stop a payment before it happens.

Should it use **online** or **batch** prediction?
**Explain why.**

### LEB2 Submission

A PDF file with complete answers for all questions.
Output predictions.csv file

