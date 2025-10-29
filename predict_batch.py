
import argparse, time, joblib, pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input CSV with a 'hours' column")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV with predictions")
    args = parser.parse_args()

    model = joblib.load("model.joblib")
    df = pd.read_csv(args.input)

    t0 = time.time()
    proba = model.predict_proba(df[["hours"]])[:,1]
    pred = (proba >= 0.5).astype(int)
    t1 = time.time()

    df_out = df.copy()
    df_out["pass_prob"] = proba
    df_out["prediction"] = pred
    df_out.to_csv(args.output, index=False)

    total_ms = (t1 - t0) * 1000
    n = len(df_out)
    eps = n / (t1 - t0) if (t1 - t0) > 0 else float("inf")
    print(f"Wrote {n} predictions to {args.output}. Total compute time={total_ms:.2f} ms (~{eps:.1f} preds/sec)")

if __name__ == "__main__":
    main()
