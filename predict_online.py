
import argparse, time, joblib
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, required=True, help="Study hours (e.g., 5)")
    args = parser.parse_args()

    model = joblib.load("model.joblib")

    # Simulate per-request timing (no batch)
    t0 = time.time()
    proba = model.predict_proba([[args.hours]])[0,1]
    pred = int(proba >= 0.5)
    t1 = time.time()

    latency_ms = (t1 - t0) * 1000
    print(f"Hours={args.hours:.1f} -> pass_prob={proba:.3f}, prediction={pred}, latency={latency_ms:.2f} ms")

if __name__ == "__main__":
    main()
