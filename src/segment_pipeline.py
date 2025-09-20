import pandas as pd
import numpy as np

# --- Simulate dataset ---
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

# --- Segmentation logic ---
def build_segments(universe: pd.DataFrame):
    segments = {}

    def assign_segment(row):
        # AOV split
        if row["avg_order_value"] > 3000:
            aov = "HighAOV"
        elif row["avg_order_value"] > 1000:
            aov = "MidAOV"
        else:
            aov = "LowAOV"

        # Engagement split
        if row["engagement_score"] > 0.6:
            eng = "HighEng"
        elif row["engagement_score"] > 0.3:
            eng = "MidEng"
        else:
            eng = "LowEng"

        # Profitability split
        if row["profitability_score"] > 0.7:
            prof = "HighProf"
        else:
            prof = "LowProf"

        return f"{aov}_{eng}_{prof}"

    # Assign segment name to every row
    universe = universe.copy()
    universe["segment"] = universe.apply(assign_segment, axis=1)

    # --- Add ELSE bucket for any missing values ---
    universe["segment"] = universe["segment"].fillna("Other_ELSE")

    # Group into dict of segments
    for seg, df_seg in universe.groupby("segment"):
        segments[seg] = df_seg

    return segments, universe

# --- Validate MECE ---
def validate_mece(segments, universe):
    union_size = sum(len(df) for df in segments.values())
    uni_size = len(universe)
    if union_size != uni_size:
        return False, f"Union size {union_size} != universe size {uni_size}"
    return True, "MECE segmentation passed"

# --- Display segment summary ---
def print_segment_summary(segments):
    print("\nSegment Summary:")
    print("{:<30} {:>10}".format("Segment Name", "Count"))
    print("-" * 42)
    for name, df_seg in sorted(segments.items()):
        print("{:<30} {:>10}".format(name, len(df_seg)))
    print("-" * 42)
    total = sum(len(df_seg) for df_seg in segments.values())
    print("{:<30} {:>10}".format("Total Users", total))
    print()

# --- Main for manual run ---
def main():
    df = simulate_dataset(10000)
    df["days_since_abandon"] = (
        pd.Timestamp.now().normalize() - df["cart_abandoned_date"]
    ).dt.days
    universe = df[df["days_since_abandon"] <= 7].reset_index(drop=True)

    segments, uni = build_segments(universe)
    ok, msg = validate_mece(segments, uni)
    print(msg)
    print_segment_summary(segments)

if __name__ == "__main__":
    main()
