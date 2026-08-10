from src.data.synthetic_grid_data import generate_renewable_timeseries
from src.features.engineering import (
    add_calendar_cyclical_features,
    add_lag_features,
    add_rolling_features,
    create_feature_pipeline,
)


def test_calendar_cyclical_features():
    df = generate_renewable_timeseries(n_hours=100)
    feat_df = add_calendar_cyclical_features(df)
    assert "hour" in feat_df.columns
    assert "sin_hour" in feat_df.columns
    assert "cos_hour" in feat_df.columns
    assert (-1.0 <= feat_df["sin_hour"]).all() and (feat_df["sin_hour"] <= 1.0).all()


def test_lag_features():
    df = generate_renewable_timeseries(n_hours=100)
    feat_df = add_lag_features(df, target_columns=["total_renewable_mw"], lags=[1, 24])
    assert "total_renewable_mw_lag_1" in feat_df.columns
    assert "total_renewable_mw_lag_24" in feat_df.columns
    # Verify lag 1 is equal to shifted target
    assert feat_df["total_renewable_mw_lag_1"].iloc[10] == df["total_renewable_mw"].iloc[9]


def test_rolling_features_no_leakage():
    """Verify rolling features at time t use ONLY historical observations up to t-1."""
    df = generate_renewable_timeseries(n_hours=100)

    df_modified = df.copy()
    df_modified.loc[10, "total_renewable_mw"] = 99999.0

    feat_orig = add_rolling_features(df, target_columns=["total_renewable_mw"], windows=[6])
    feat_mod = add_rolling_features(df_modified, target_columns=["total_renewable_mw"], windows=[6])

    # Rolling mean at row 10 should NOT change because it uses shifted values up to row 9!
    assert feat_orig.loc[10, "total_renewable_mw_roll_6_mean"] == feat_mod.loc[10, "total_renewable_mw_roll_6_mean"]
    # Rolling mean at row 11 SHOULD change because row 10 is in history for row 11
    assert feat_orig.loc[11, "total_renewable_mw_roll_6_mean"] != feat_mod.loc[11, "total_renewable_mw_roll_6_mean"]


def test_feature_pipeline():
    df = generate_renewable_timeseries(n_hours=500)
    feat_df = create_feature_pipeline(df, drop_na=True)
    assert len(feat_df) < 500  # Dropped lag NaNs
    assert len(feat_df) > 300
    assert not feat_df.isna().any().any()
