"""Candidate regressors (FR07). XGBoost is used when installed, otherwise sklearn's
HistGradientBoostingRegressor is used as a clearly labelled stand-in."""
from __future__ import annotations
import importlib.util
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


def xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def build_candidates(cfg: dict) -> tuple[dict, dict]:
    seed = cfg.get("seed", 42)
    m = cfg["models"]
    rf = RandomForestRegressor(random_state=seed, n_jobs=-1, **m["random_forest"])
    cands = {"Random Forest": rf}
    notes = {"Random Forest": "sklearn.ensemble.RandomForestRegressor"}
    x = m["xgboost"]
    if xgboost_available():
        from xgboost import XGBRegressor
        cands["XGBoost"] = XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=4, **x)
        notes["XGBoost"] = "xgboost.XGBRegressor"
    else:
        cands["XGBoost"] = HistGradientBoostingRegressor(
            max_iter=x["n_estimators"], learning_rate=x["learning_rate"], max_depth=x["max_depth"],
            random_state=seed)
        notes["XGBoost"] = "FALLBACK: sklearn HistGradientBoostingRegressor (xgboost not installed)"
    return cands, notes
