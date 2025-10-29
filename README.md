
# ML Model Serving: Online vs Batch Prediction

This tiny lab shows the difference between **online** (predict per request) and **batch** (predict many at once) for a simple model.

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

---

## **Part 1 — Train the Model**

Run:

```bash
python train_model.py
```

### Q1:

What accuracy does the model show on the test set?

*Answer:*

```
____________________
```

---

## **Part 2 — Online Prediction**

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

### Q3:

Why is latency reported for every prediction?

*Answer:*

```
________________________________________________________
________________________________________________________
```

---

## **Part 3 — Batch Prediction**

Run:

```bash
python predict_batch.py --input data/batch_inputs.csv --output predictions.csv
```

### Q4:

How many rows were processed?

```
Number of rows: ____________
```

### Q5:

What was the **total compute time** and **predictions per second**?

```
Total time: __________ ms
Throughput: __________ predictions/sec
```

---

## **Part 4 — Compare Online vs Batch**

| Mode   | When is result available? | Efficiency | When is it useful? |
| ------ | ------------------------- | ---------- | ------------------ |
| Online |                           |            |                    |
| Batch  |                           |            |                    |

### Q6:

In your own words:
**Why is batch mode faster when predicting many rows?**

```
________________________________________________________
________________________________________________________
```

---

## **Part 5 — Critical Thinking**

Imagine a **fraud detection system** that must stop a payment before it happens.

Should it use **online** or **batch** prediction?
**Explain why.**

```
________________________________________________________
________________________________________________________
```

