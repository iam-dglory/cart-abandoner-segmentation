# src/segment_pipeline.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def simulate_dataset(n=10000, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        "user_id": np.arange(1, n+1),
        "cart_abandoned_date": pd.Timestamp.now().normalize() 
            - pd.to_timedelta(np.random.randint(0, 15, size=n), unit="d"),
        "last_order_date": pd.Timestamp.now().normalize() 
            - pd.to_timedelta(np.random.randint(0, 60, size=n), unit="d"),
        "avg_order_value": np.random.randint(100, 5000, size=n),
        "sessions_last_30d": np.random.poisson(lam=5, size=n),
        "num_cart_items": np.random.randint(1, 10, size=n),
        "engagement_score": np.random.rand(n),
        "profitability_score": np.random.rand(n),
    })
    return df

def build_segments(universe: pd.DataFrame):
    segments = {}

    def assign_segment(row):
        # --- AOV ---
        if row["avg_order_value"] > 3000:
            aov = "HighAOV"
        elif row["avg_order_value"] > 1000:
            aov = "MidAOV"
        else:
            aov = "LowAOV"

        # --- Engagement ---
        if row["engagement_score"] > 0.6:
            eng = "HighEng"
        elif row["engagement_score"] > 0.3:
            eng = "MidEng"
        else:
            eng = "LowEng"

        # --- Profitability ---
        if row["profitability_score"] > 0.7:
            prof = "HighProf"
        else:
            prof = "LowProf"

        return f"{aov}_{eng}_{prof}"

    universe = universe.copy()
    universe["segment"] = universe.apply(assign_segment, axis=1)
    universe["segment"] = universe["segment"].fillna("Other_ELSE")

    # Group by segment
    for seg, df_seg in universe.groupby("segment"):
        segments[seg] = df_seg

    return segments, universe

def validate_mece(segments, universe):
    union_size = sum(len(df) for df in segments.values())
    uni_size = len(universe)
    if union_size != uni_size:
        return False, f"Union size {union_size} != universe size {uni_size}"
    return True, "MECE segmentation passed"

def compute_scores(segments):
    scores = {}
    total_users = sum(len(df) for df in segments.values())
    for name, df_seg in segments.items():
        conv_pot = df_seg['engagement_score'].mean()
        profitability = df_seg['profitability_score'].mean()
        lift_vs_control = np.random.rand()  # simulate A/B lift
        size_norm = len(df_seg) / total_users
        overall = 0.4*conv_pot + 0.4*profitability + 0.1*lift_vs_control + 0.1*size_norm
        scores[name] = {
            'conv_pot': conv_pot,
            'profitability': profitability,
            'lift_vs_control': lift_vs_control,
            'size_norm': size_norm,
            'overall': overall,
            'size': len(df_seg),
            'valid': True
        }
    # Rerank by overall score
    scores = dict(sorted(scores.items(), key=lambda x: x[1]['overall'], reverse=True))
    return scores

def print_segment_summary(segments, scores):
    print("\nSegment Summary:")
    print("{:<35} {:>6} {:>10}".format("Segment Name", "Size", "Overall Score"))
    print("-" * 55)
    for name, s in scores.items():
        print("{:<35} {:>6} {:>10.2f}".format(name, s['size'], s['overall']))
    total = sum(len(df_seg) for df_seg in segments.values())
    print("-" * 55)
    print("{:<35} {:>6}".format("Total Users", total))
    print()
    
def export_outputs(scores):
    os.makedirs("output", exist_ok=True)
    # CSV
    df_out = pd.DataFrame([{
        "Segment Name": name,
        "Size": s['size'],
        "Conv_Pot": round(s['conv_pot'],2),
        "Profitability": round(s['profitability'],2),
        "Overall Score": round(s['overall'],2),
        "Valid": s['valid']
    } for name,s in scores.items()])
    df_out.to_csv("output/segments_summary.csv", index=False)
    # JSON
    import json
    with open("output/segments_summary.json", "w") as f:
        json.dump(scores, f, indent=4)

def plot_segments(segments):
    sizes = [len(s) for s in segments.values()]
    labels = list(segments.keys())
    plt.figure(figsize=(10,6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Cart Abandoner Segment Distribution")
    plt.tight_layout()
    plt.show()

def main():
    df = simulate_dataset(10000)
    df["days_since_abandon"] = (pd.Timestamp.now().normalize() - df["cart_abandoned_date"]).dt.days
    universe = df[df["days_since_abandon"] <= 7].reset_index(drop=True)

    segments, uni = build_segments(universe)
    ok, msg = validate_mece(segments, uni)
    print(msg)

    scores = compute_scores(segments)
    print_segment_summary(segments, scores)
    export_outputs(scores)
    plot_segments(segments)  # optional

if __name__ == "__main__":
    main()
