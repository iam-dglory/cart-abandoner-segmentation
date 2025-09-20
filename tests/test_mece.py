import pandas as pd
from src.segment_pipeline import main, build_segments, validate_mece, simulate_dataset
def test_mece_on_simulated_small():
    df = simulate_dataset(n=5000)
    df["cart_abandoned_date"] = pd.to_datetime(df["cart_abandoned_date"])
    df["days_since_abandon"] = (pd.Timestamp.now().normalize() - df["cart_abandoned_date"]).dt.days
    universe = df[df["days_since_abandon"] <= 7].reset_index(drop=True)
    segments, uni = build_segments(universe)
    ok, msg = validate_mece(segments, uni)
    assert ok, msg
