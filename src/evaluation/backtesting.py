"""Temporal cross-validation and walk-forward backtesting framework."""

from typing import Dict, Any, List, Optional, Type
import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_wape, pinball_loss, crps_score


class WalkForwardBacktester:
    """Walk-Forward Backtesting Framework for temporal time series evaluation.

    Simulates realistic production forecasting by sequentially expanding or rolling
    the training window and evaluating on non-overlapping future horizons.
    """

    def __init__(
        self,
        train_window_hours: int = 336,
        test_window_hours: int = 24,
        step_hours: int = 24,
        min_train_hours: int = 168,
        expanding: bool = True,
    ):
        self.train_window_hours = train_window_hours
        self.test_window_hours = test_window_hours
        self.step_hours = step_hours
        self.min_train_hours = min_train_hours
        self.expanding = expanding

    def backtest_tabular_model(
        self,
        model_factory,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "total_renewable_mw",
        time_col: str = "timestamp",
    ) -> Dict[str, Any]:
        """Execute walk-forward backtest for tabular ML forecasters (XGBoost, LightGBM).

        Parameters
        ----------
        model_factory : callable
            Function returning a fresh un-fitted model instance.
        df : pd.DataFrame
            Feature-augmented DataFrame sorted chronologically.
        feature_cols : list of str
            Input feature names.
        target_col : str, default='total_renewable_mw'
            Target column.
        time_col : str, default='timestamp'
            Timestamp column.

        Returns
        -------
        dict
            Backtest evaluation summary metrics and fold predictions.
        """
        data = df.sort_values(time_col).reset_index(drop=True)
        n = len(data)

        if n < self.min_train_hours + self.test_window_hours:
            raise ValueError(f"Insufficient dataset length ({n} rows) for backtesting.")

        fold_metrics: List[Dict[str, float]] = []
        all_actuals = []
        all_p10 = []
        all_p50 = []
        all_p90 = []
        all_dates = []

        start_idx = self.train_window_hours
        cutoff_indices = list(range(start_idx, n - self.test_window_hours + 1, self.step_hours))

        for cutoff in cutoff_indices:
            if self.expanding:
                train_df = data.iloc[:cutoff]
            else:
                train_df = data.iloc[max(0, cutoff - self.train_window_hours) : cutoff]

            test_df = data.iloc[cutoff : cutoff + self.test_window_hours]

            X_train, y_train = train_df[feature_cols], train_df[target_col]
            X_test, y_test = test_df[feature_cols], test_df[target_col].values

            # Fit fresh model instance
            model = model_factory()
            model.fit(X_train, y_train)

            # Predict quantiles
            preds = model.predict_quantiles(X_test)
            p10 = preds.get("p10", preds["p50"] - 5.0)
            p50 = preds["p50"]
            p90 = preds.get("p90", preds["p50"] + 5.0)

            # Compute fold metrics
            mae = float(np.mean(np.abs(y_test - p50)))
            rmse = float(np.sqrt(np.mean((y_test - p50) ** 2)))
            wape = calculate_wape(y_test, p50)
            crps = crps_score(y_test, p10, p50, p90)

            fold_metrics.append({
                "cutoff": str(data.iloc[cutoff][time_col]),
                "mae": mae,
                "rmse": rmse,
                "wape": wape,
                "crps": crps,
            })

            all_actuals.extend(y_test)
            all_p10.extend(p10)
            all_p50.extend(p50)
            all_p90.extend(p90)
            all_dates.extend(test_df[time_col].values)

        actuals_arr = np.array(all_actuals)
        p10_arr = np.array(all_p10)
        p50_arr = np.array(all_p50)
        p90_arr = np.array(all_p90)

        mean_mae = float(np.mean([m["mae"] for m in fold_metrics]))
        mean_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
        mean_wape = float(np.mean([m["wape"] for m in fold_metrics]))
        mean_crps = float(np.mean([m["crps"] for m in fold_metrics]))

        return {
            "n_windows": len(fold_metrics),
            "mean_mae": round(mean_mae, 4),
            "mean_rmse": round(mean_rmse, 4),
            "mean_wape": round(mean_wape, 4),
            "mean_crps": round(mean_crps, 4),
            "fold_metrics": fold_metrics,
            "actuals": actuals_arr,
            "p10": p10_arr,
            "p50": p50_arr,
            "p90": p90_arr,
            "timestamps": all_dates,
        }
