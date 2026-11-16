"""GridSense — AI-Powered Electricity Consumption Forecasting & Visualisation System.

A SaaS-style smart grid demand intelligence dashboard with:
- Modern SaaS light theme
- 6 Navigation pages: Overview, Historical Analysis, Forecasting, Model Performance, Data Quality, About Project
- 5 Forecasting models: 3 Baselines + Random Forest + XGBoost
- Leakage-free chronological evaluation
- Next-hour forecasting, recursive multi-step forecasting, and 3-tier demand classification
- Interactive Plotly visualizations and Phase 6.1 compliant CSV exports
"""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from src.config import dataset_path, load_config, model_dir, processed_path, report_dir
from src.data_loader import PORTFOLIO, DataError, list_clients
from src.evaluate import classify_demand
from src.forecast import (
    ForecastError,
    artefacts_exist,
    forecast_next_hour,
    format_export_csv,
    load_artefacts,
    recursive_forecast,
)
from src.pipeline import run_experiment
from ui import charts
from ui.theme import (
    brand,
    card,
    demand_status_badge,
    disclaimer,
    hero,
    inject_css,
    kpi,
    kpi_row,
    section,
)

LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.jpg"
st.set_page_config(
    page_title="GridSense · AI Electricity Forecasting",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

CFG = load_config()
try:
    DATA_PATH = dataset_path(CFG)
    DATA_PATH_STR = str(DATA_PATH)
    DATA_MTIME = DATA_PATH.stat().st_mtime
    DATA_AVAILABLE = True
except FileNotFoundError:
    DATA_PATH = None
    DATA_PATH_STR = ""
    DATA_MTIME = 0.0
    DATA_AVAILABLE = False
IS_SYNTH = bool(CFG.get("dataset", {}).get("is_synthetic", False))

PAGES = [
    "Overview",
    "Historical Analysis",
    "Forecasting",
    "Model Performance",
    "Data Quality",
    "About Project",
]

ICONS = {
    "Overview": "📊",
    "Historical Analysis": "📈",
    "Forecasting": "⚡",
    "Model Performance": "🏆",
    "Data Quality": "🛡️",
    "About Project": "📖",
}


# ----------------------------------------------------------------------------- Data Caching Helpers
def _stamp(client: str) -> int:
    p = model_dir(client) / "metadata.json"
    return p.stat().st_mtime_ns if p.exists() else 0


@st.cache_data(show_spinner=False)
def get_clients(path: str, mtime: float) -> list[str]:
    rep = CFG.get("dataset", {}).get("representative_clients")
    if rep:
        return list(rep) + [PORTFOLIO]
    if path and Path(path).exists():
        return list_clients(path) + [PORTFOLIO]
    return [PORTFOLIO]


@st.cache_resource(show_spinner=False)
def get_model(client: str, stamp: int):
    return load_artefacts(model_dir(client))


@st.cache_data(show_spinner=False)
def get_tables(client: str, stamp: int) -> dict:
    rd = report_dir(client)
    hourly = pd.read_csv(processed_path(client), index_col=0, parse_dates=True)
    out = {
        "hourly": hourly,
        "preds": pd.read_csv(rd / "test_predictions.csv", index_col=0, parse_dates=True),
        "metrics": pd.read_csv(rd / "metrics.csv"),
        "importance": pd.read_csv(rd / "feature_importance.csv"),
        "by_hour": pd.read_csv(rd / "error_by_hour.csv"),
        "quality": json.loads((rd / "data_quality.json").read_text(encoding="utf-8")),
    }
    return out


@st.cache_data(show_spinner=False)
def get_forecast(client: str, stamp: int, origin_iso: str | None, horizon: int) -> pd.DataFrame:
    model, meta = get_model(client, stamp)
    y = get_tables(client, stamp)["hourly"]["kwh"]
    hist = y if origin_iso is None else y.loc[: pd.Timestamp(origin_iso)]
    return recursive_forecast(model, hist, horizon=horizon, meta=meta, client=client)


def fmt(x, nd=2, unit=""):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{nd}f}{unit}"


# ----------------------------------------------------------------------------- Training View
def train_view(client: str) -> None:
    hero(
        f"Train AI Forecasting Models for {client}",
        "No trained model was found for this series yet. GridSense will load raw 15-minute kW readings, "
        "convert them into hourly kWh, engineer leakage-free features, train Random Forest and XGBoost candidates, "
        "score against 3 seasonal baselines, and evaluate all 5 models on an untouched chronological test period.",
        [("Initial Setup", "amber")],
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        card("1 · Data Preprocessing", "<p>15-min kW → Hourly kWh<br>European DST handling<br>Structural zero trimming</p>")
    with col2:
        card("2 · Chronological ML", "<p>Persistence & Seasonal Baselines<br>Random Forest & XGBoost<br>70/15/15 Chronological Split</p>")
    with col3:
        card("3 · Transparent Evaluation", "<p>All 5 models scored on test<br>MAE, RMSE, ME bias<br>Statistical High-Demand Alerts</p>")

    st.write("")
    if st.button("🚀  Train Model Pipeline Now", type="primary", key=f"train_{client}"):
        bar = st.progress(0.0, text="Starting training pipeline...")

        def cb(frac: float, msg: str) -> None:
            bar.progress(min(max(frac, 0.0), 1.0), text=msg)

        try:
            run_experiment(client, CFG, progress=cb)
        except (DataError, ValueError, ForecastError) as exc:
            st.error(f"Training failed: {exc}")
            return
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.caption("Takes roughly 10–40 seconds depending on your machine. You can also run `python -m src.pipeline --client "
               f"{client}` from your terminal.")


# ----------------------------------------------------------------------------- PAGE 1: Overview
def page_overview(client, meta, T, model_stamp):
    hourly = T["hourly"]
    y = hourly["kwh"]
    p25 = meta.get("p25_threshold_kwh", float(np.percentile(y, 25)))
    p95 = meta.get("alert_threshold_kwh", float(np.percentile(y, 95)))
    sel = meta.get("selected_model", "XGBoost")

    fc = get_forecast(client, model_stamp, None, 24)
    last_ts = y.index[-1]
    last_val = float(y.iloc[-1])
    nxt_val = float(fc["predicted_kwh"].iloc[0])
    peak_fc = float(fc["predicted_kwh"].max())
    n_alerts = int(fc["alert"].sum())

    # Latest observed demand tier
    obs_tier, obs_exp, _ = classify_demand(last_val, p25, p95)
    fc_tier, fc_exp, is_fc_alert = classify_demand(nxt_val, p25, p95)

    pills = [
        (f"Series: {client}", "blue"),
        (f"Selected Model: {sel}", "emerald"),
        ("Synthetic Demo Data", "amber") if IS_SYNTH else ("Real UCI Dataset", "emerald"),
    ]
    hero(
        "Electricity Demand Intelligence Overview",
        f"Demand intelligence and next-hour forecasting for series {client}. Historical observations available up to "
        f"{last_ts:%a, %d %b %Y, %H:%M}. Forecasts generated using chronologically validated machine learning models.",
        pills,
    )

    series_desc = "Aggregate of 370 Observed Clients" if client == PORTFOLIO else ("Synthetic Client Series" if IS_SYNTH else "Individual Client Series")
    kpi_row([
        ("Selected Series", client, "", series_desc, "blue"),
        ("Valid Hourly Records", f"{len(y):,}", "hours", f"from {y.index[0]:%d %b %Y}", "indigo"),
        ("Average Energy", fmt(y.mean(), 1), "kWh", f"Std Dev: {y.std():.1f} kWh", "slate"),
        ("Peak Historical Energy", fmt(y.max(), 1), "kWh", f"at {y.idxmax():%d %b %H:%M}", "amber"),
        ("Latest Observed Hour", fmt(last_val, 1), "kWh", f"ended {last_ts:%d %b %H:%M}", "blue"),
        ("Next-Hour Forecast", fmt(nxt_val, 1), "kWh", f"target {fc['target_time'].iloc[0]:%H:%M}", "indigo"),
        ("Selected Model", sel, "", f"best validation MAE", "emerald"),
    ])

    st.write("")
    demand_status_badge(fc_tier, f"Next-Hour Forecast ({nxt_val:.1f} kWh): {fc_exp}")

    # Historical consumption chart
    section("Historical Consumption Overview")
    st.plotly_chart(charts.history_chart(y, p95, "Hourly Energy Consumption (kWh)"), use_container_width=True, key="ov_history")

    # Load shapes
    col_a, col_b = st.columns(2)
    with col_a:
        section("Diurnal Load Pattern")
        st.plotly_chart(charts.hourly_profile(y), use_container_width=True, key="ov_diurnal")
    with col_b:
        section("Weekly Demand Distribution")
        st.plotly_chart(charts.weekday_bars(y), use_container_width=True, key="ov_weekday")

    # Bottom Insights and Peak Watchlist
    left, right = st.columns([1.1, 1])
    with left:
        section("24-Hour Peak Demand Watchlist")
        alerts = fc[fc["alert"]]
        if alerts.empty:
            st.markdown(
                f'<div class="gs-status-banner low">'
                f'<span>✓ <strong>All Clear:</strong> No forecast hour exceeds the statistical P95 threshold '
                f'({p95:,.1f} kWh) in the next 24 hours.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            for _, r in alerts.iterrows():
                st.markdown(
                    f'<div class="gs-status-banner high">'
                    f'<span>⚠ <strong>{r["target_time"]:%a %d %b · %H:%M}</strong> — '
                    f'Predicted {r["predicted_kwh"]:,.1f} kWh '
                    f'({r.get("lower_kwh", 0):,.0f}–{r.get("upper_kwh", 0):,.0f} kWh)</span></div>',
                    unsafe_allow_html=True,
                )
        st.caption("Forecast high-demand alerts flag predicted energy exceeding the historical training P95 baseline. "
                   "They represent statistical demand anomalies, not physical grid equipment failures.")

    with right:
        section("Key Dataset Insights")
        prof = y.groupby(y.index.hour).mean()
        wk, we = y[y.index.dayofweek < 5].mean(), y[y.index.dayofweek >= 5].mean()
        above_pct = float((y > p95).mean() * 100)
        items = [
            f"Busiest hour on average: <strong>{int(prof.idxmax()):02d}:00</strong> ({prof.max():,.1f} kWh); quietest: <strong>{int(prof.idxmin()):02d}:00</strong> ({prof.min():,.1f} kWh).",
            f"Weekday energy consumption is <strong>{wk / we:.2f}×</strong> weekend consumption." if we else "",
            f"<strong>{above_pct:.1f}%</strong> of historical observations exceed the P95 alert threshold ({p95:,.1f} kWh).",
            f"Selected forecasting model: <strong>{sel}</strong> with empirical prediction intervals.",
        ]
        card("Automated Load Summary", "<ul>" + "".join(f"<li>{i}</li>" for i in items if i) + "</ul>")


# ----------------------------------------------------------------------------- PAGE 2: Historical Analysis
def page_historical(client, meta, T, model_stamp):
    y = T["hourly"]["kwh"]
    p25 = meta.get("p25_threshold_kwh", float(np.percentile(y, 25)))
    p95 = meta.get("alert_threshold_kwh", float(np.percentile(y, 95)))

    hero(
        "Historical Electricity Consumption Analysis",
        "Deep-dive into historical load patterns, seasonality, day-of-week trends, and peak consumption periods. "
        "Filter by date range to inspect specific historical intervals.",
        [("Series: " + client, "blue"), ("Granularity: Hourly kWh", "indigo")],
    )

    section("Date Range Filter")
    d_min = y.index.min().date()
    d_max = y.index.max().date()
    c_f1, c_f2 = st.columns(2)
    start_date = c_f1.date_input("Start Date", value=d_min, min_value=d_min, max_value=d_max, key="hist_start")
    end_date = c_f2.date_input("End Date", value=d_max, min_value=d_min, max_value=d_max, key="hist_end")

    if start_date > end_date:
        st.error("Start date must be before or equal to End date.")
        return

    y_sub = y.loc[str(start_date): str(end_date)]
    if y_sub.empty:
        st.warning("No data points available for the selected date range.")
        return

    kpi_row([
        ("Observations in Range", f"{len(y_sub):,}", "hours", f"{start_date} to {end_date}", "blue"),
        ("Total Energy in Range", f"{y_sub.sum():,.1f}", "kWh", f"Sum of hourly energy", "indigo"),
        ("Average Energy", fmt(y_sub.mean(), 2), "kWh", f"Median: {y_sub.median():.2f} kWh", "emerald"),
        ("Peak Demand in Range", fmt(y_sub.max(), 1), "kWh", f"{y_sub.idxmax():%d %b %Y %H:%M}", "rose"),
    ])

    tab1, tab2, tab3 = st.tabs(["Consumption Trend", "Diurnal & Calendar Patterns", "Peak Periods & Distribution"])

    with tab1:
        res = st.radio("Aggregation Level", ["Hourly", "Daily Sum", "Monthly Sum"], horizontal=True, key="hist_res")
        if res == "Hourly":
            s_plot, lab = y_sub, "Hourly Energy Consumption (kWh)"
            st.plotly_chart(charts.history_chart(s_plot, p95, lab), use_container_width=True, key="hist_hourly")
        elif res == "Daily Sum":
            s_plot = y_sub.resample("D").sum()
            lab = "Daily Total Energy (kWh)"
            st.plotly_chart(charts.history_chart(s_plot, None, lab), use_container_width=True, key="hist_daily")
        else:
            try:
                s_plot = y_sub.resample("ME").sum()
            except ValueError:
                s_plot = y_sub.resample("M").sum()
            lab = "Monthly Total Energy (kWh)"
            st.plotly_chart(charts.history_chart(s_plot, None, lab), use_container_width=True, key="hist_monthly")

    with tab2:
        st.plotly_chart(charts.heatmap_hour_weekday(y_sub), use_container_width=True, key="hist_heatmap")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.hourly_profile(y_sub), use_container_width=True, key="hist_profile")
        with c2:
            st.plotly_chart(charts.monthly_bars(y_sub), use_container_width=True, key="hist_monthly_bar")

    with tab3:
        st.plotly_chart(charts.distribution(y_sub, p25=p25, p95=p95), use_container_width=True, key="hist_dist")
        section("Top 10 Highest Consumption Peaks in Selected Range")
        peaks = y_sub.nlargest(10).to_frame(name="Energy (kWh)").reset_index()
        ts_col = peaks.columns[0]
        peaks["Timestamp"] = pd.to_datetime(peaks[ts_col]).dt.strftime("%Y-%m-%d %H:%M")
        peaks["Demand Tier"] = ["HIGH" if val > p95 else "NORMAL" for val in peaks["Energy (kWh)"]]
        st.dataframe(peaks[["Timestamp", "Energy (kWh)", "Demand Tier"]], hide_index=True)


# ----------------------------------------------------------------------------- PAGE 3: Forecasting
def page_forecasting(client, meta, T, model_stamp):
    hourly = T["hourly"]
    y = hourly["kwh"]
    sel = meta.get("selected_model", "XGBoost")
    p25 = meta.get("p25_threshold_kwh", float(np.percentile(y, 25)))
    p95 = meta.get("alert_threshold_kwh", float(np.percentile(y, 95)))
    last_obs_time = y.index[-1]
    last_obs_val = float(y.iloc[-1])

    hero(
        "Next-Hour & Multi-Step Electricity Demand Forecasting",
        "Generate AI-powered forecasts relative to the latest available dataset observations. "
        "Strictly leakage-free: predictions are generated using only historical observations available at the issue time.",
        [("Selected Model: " + sel, "emerald"), ("Historical Dataset Mode", "amber")],
    )

    st.info(
        "ℹ️ **Dataset Boundary Notice:** This application operates on historical smart meter data. "
        f"The latest recorded observation for series **{client}** was recorded at **{last_obs_time:%a, %d %b %Y, %H:%M}**. "
        "The next-hour forecast is generated for the subsequent hour ending at the target timestamp, not the current real-world clock time."
    )

    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col1:
        st.markdown(f"**Forecasting Model:** `{sel}` (Validation-Selected)")
        st.markdown(f"**Latest Observation:** `{last_obs_val:.2f} kWh` (ended {last_obs_time:%d %b %Y, %H:%M})")
    with col2:
        horizon = st.slider("Forecast Horizon (Hours)", min_value=1, max_value=48, value=24, step=1, key="fc_horizon")
    with col3:
        show_intervals = st.toggle("Show Prediction Intervals", value=True, key="fc_toggle_int")

    st.write("")
    gen_clicked = st.button("🚀  GENERATE FORECAST", type="primary", use_container_width=True, key="btn_gen_forecast")

    cache_key = f"cached_fc_{client}_{model_stamp}"
    if gen_clicked:
        with st.spinner(f"Computing {horizon}-hour electricity demand forecast for series {client} using {sel}..."):
            try:
                fc_df = get_forecast(client, model_stamp, None, horizon)
                st.session_state[cache_key] = {"horizon": horizon, "df": fc_df}
            except Exception as exc:
                st.error(f"Failed to generate forecast: {exc}")
                return

    cached_data = st.session_state.get(cache_key)
    if cached_data is None:
        st.markdown(
            f'<div class="gs-card" style="text-align: center; padding: 2.8rem 1.5rem; margin-top: 1.5rem;">'
            f'<div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚡</div>'
            f'<h3 style="margin-top: 0; color: #0F172A; font-size: 1.35rem;">Forecasting Control Room Ready</h3>'
            f'<p style="color: #64748B; max-width: 620px; margin: 0.5rem auto 1.2rem auto; font-size: 0.95rem; line-height: 1.6;">'
            f'The forecasting engine is primed for series <strong>{client}</strong> with validation-selected model '
            f'<strong>{sel}</strong>. Configure your forecast horizon (1–48 hours) above and click '
            f'<strong>🚀 GENERATE FORECAST</strong> to compute next-hour predictions, multi-step projections, and statistical demand alerts.'
            f'</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    fc_df = cached_data["df"]
    if cached_data["horizon"] != horizon:
        st.info(f"ℹ️ Currently displaying cached **{cached_data['horizon']}-hour** forecast. Click **🚀 GENERATE FORECAST** above to compute for **{horizon} hours**.")

    nxt_row = fc_df.iloc[0]
    pred_nxt = float(nxt_row["predicted_kwh"])
    tier, exp, is_alert = classify_demand(pred_nxt, p25, p95)

    # Next-Hour Forecast Hero Card
    st.markdown(
        f'<div class="gs-prediction-card">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">'
        f'<div>'
        f'<div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; color:#64748B; letter-spacing:0.08em;">Next-Hour Target Forecast</div>'
        f'<div style="font-size:2.4rem; font-weight:800; color:#0F172A; margin:0.2rem 0;">{pred_nxt:,.2f} <span style="font-size:1.1rem; color:#64748B; font-weight:600;">kWh</span></div>'
        f'<div style="font-size:0.88rem; color:#475569;">Target Hour: <strong>{nxt_row["target_time"]:%a %d %b %Y, %H:%M}</strong> (Issue time: {nxt_row["issue_time"]:%d %b %H:%M})</div>'
        f'</div>'
        f'<div>'
        f'<span class="gs-pill {tier.lower()}" style="font-size:0.95rem; padding:0.4rem 1rem;">Demand Status: {tier}</span>'
        f'</div>'
        f'</div>'
        f'<div class="gs-prediction-grid">'
        f'<div><span style="color:#64748B; font-size:0.75rem; text-transform:uppercase; font-weight:700;">Client Series</span><br><strong>{client}</strong></div>'
        f'<div><span style="color:#64748B; font-size:0.75rem; text-transform:uppercase; font-weight:700;">Forecasting Engine</span><br><strong>{sel}</strong></div>'
        f'<div><span style="color:#64748B; font-size:0.75rem; text-transform:uppercase; font-weight:700;">Historical Baseline</span><br><strong>{last_obs_val:.2f} kWh</strong></div>'
        f'<div><span style="color:#64748B; font-size:0.75rem; text-transform:uppercase; font-weight:700;">Alert Threshold (P95)</span><br><strong>{p95:.1f} kWh</strong></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    demand_status_badge(tier, exp)

    section("Forecast Fan Chart")
    st.caption("ℹ️ **Methodology Notice:** The primary benchmarked and validated model objective is next-hour (1-hour-ahead) forecasting. Horizons beyond 1 hour (2–48h) are generated recursively using predicted autoregressive lags as an experimental exploratory projection with empirical prediction intervals.")
    hist_tail = y.iloc[-24 * 4:]
    st.plotly_chart(
        charts.forecast_fan(hist_tail, fc_df, threshold=p95, show_interval=show_intervals),
        use_container_width=True,
        key="fc_fan_chart",
    )

    section("Forecast Schedule Table")
    display_cols = ["target_time", "forecast_horizon_hours", "predicted_kwh", "alert_level", "alert_status"]
    if "lower_kwh" in fc_df.columns and "upper_kwh" in fc_df.columns:
        display_cols.extend(["lower_kwh", "upper_kwh"])

    show_tbl = fc_df[display_cols].copy()
    show_tbl["target_time"] = show_tbl["target_time"].dt.strftime("%Y-%m-%d %H:%M")
    show_tbl = show_tbl.rename(columns={
        "target_time": "Target Timestamp",
        "forecast_horizon_hours": "Horizon (Hours)",
        "predicted_kwh": "Predicted (kWh)",
        "alert_level": "Demand Tier",
        "alert_status": "Alert Status",
        "lower_kwh": "Lower Bound (kWh)",
        "upper_kwh": "Upper Bound (kWh)",
    })
    st.dataframe(show_tbl.round(2), hide_index=True)

    # Download button (Phase 6.1 compliant)
    csv_str = format_export_csv(fc_df)
    st.download_button(
        label="⬇ Download Forecast CSV",
        data=csv_str.encode("utf-8"),
        file_name=f"gridsense_forecast_{client}_{last_obs_time:%Y%m%d_%H}.csv",
        mime="text/csv",
        key="btn_download_forecast",
    )


# ----------------------------------------------------------------------------- PAGE 4: Model Performance
def page_performance(client, meta, T, model_stamp):
    metrics = T["metrics"].copy()
    preds = T["preds"].copy()
    sel = meta.get("selected_model", "XGBoost")
    best_base = meta.get("best_baseline", "Persistence")

    hero(
        "Model Performance Leaderboard & Evaluation",
        "Rigorous, transparent evaluation of all 5 forecasting models on the untouched chronological test period. "
        "Baselines and machine learning models are scored on identical timestamps.",
        [("Evaluation: Untouched Test Period", "emerald"), ("Selected Model: " + sel, "blue")],
    )

    # Best model on test
    best_test_row = metrics.loc[metrics["MAE"].idxmin()]
    kpi_row([
        ("Selected Model", sel, "", f"chosen via validation", "blue"),
        ("Lowest Test MAE", f"{best_test_row['MAE']:.3f}", "kWh", f"{best_test_row['model_name']}", "emerald"),
        ("Lowest Test RMSE", f"{metrics['RMSE'].min():.3f}", "kWh", f"{metrics.loc[metrics['RMSE'].idxmin()]['model_name']}", "indigo"),
        ("Reference Baseline", best_base, "", f"best validation baseline", "slate"),
    ])

    tab1, tab2, tab3 = st.tabs(["5-Model Leaderboard", "Actual vs. Predicted", "Error Analysis & Feature Importance"])

    with tab1:
        section("All 5 Forecasting Approaches Compared on Test Period")
        cols_to_show = ["model_name", "kind"]
        if "validation_mae" in metrics.columns:
            cols_to_show.append("validation_mae")
        cols_to_show.extend(["MAE", "RMSE", "ME", "MAE_improvement_pct", "RMSE_improvement_pct"])
        if "model_selection_status" in metrics.columns:
            cols_to_show.append("model_selection_status")
        cols_to_show.append("sample_count")

        rename_dict = {
            "model_name": "Model",
            "kind": "Type",
            "validation_mae": "Validation MAE (kWh)",
            "MAE": "Test MAE (kWh)",
            "RMSE": "Test RMSE (kWh)",
            "ME": "Test ME (kWh)",
            "MAE_improvement_pct": "MAE Impr. (%)",
            "RMSE_improvement_pct": "RMSE Impr. (%)",
            "model_selection_status": "Model-selection status",
            "sample_count": "Test Hours",
        }
        show_m = metrics[cols_to_show].rename(columns=rename_dict)
        st.dataframe(show_m, hide_index=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.plotly_chart(charts.leaderboard_mae_chart(metrics), use_container_width=True, key="lb_mae")
        with col_m2:
            st.plotly_chart(charts.leaderboard_rmse_chart(metrics), use_container_width=True, key="lb_rmse")

        st.write("")
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            card(
                "Understanding the Evaluation Metrics",
                "<ul>"
                "<li><strong>MAE (Mean Absolute Error):</strong> The average magnitude of errors in kWh. A lower value indicates better overall accuracy.</li>"
                "<li><strong>RMSE (Root Mean Squared Error):</strong> Penalises large errors more severely by squaring differences before averaging.</li>"
                "<li><strong>ME (Mean Error / Bias):</strong> Measures directional tendency. Positive ME indicates overprediction; negative ME indicates underprediction.</li>"
                "</ul>",
            )
        with c_exp2:
            card(
                "Why Chronological Validation Matters",
                "<p>Time-series observations must never be randomly shuffled. Shuffling introduces forward-looking data leakage. "
                "GridSense strictly trains on the first 70% of historical data, tunes hyperparameters and selects the winning model "
                "on the next 15% (validation), and evaluates only once on the final 15% (test).</p>",
            )

    with tab2:
        section("Time-Series Test Replay")
        window = st.select_slider(
            "Select Inspection Window",
            options=["Last 3 Days", "Last 7 Days", "Last 14 Days", "Full Test Period"],
            value="Last 7 Days",
            key="test_window_slider",
        )
        n_hours = {"Last 3 Days": 72, "Last 7 Days": 168, "Last 14 Days": 336}.get(window)
        p_sub = preds if n_hours is None else preds.iloc[-n_hours:]

        st.plotly_chart(charts.actual_vs_pred(p_sub, sel, best_base), use_container_width=True, key="perf_avp")

    with tab3:
        sc = preds[preds["scoreable"]]
        residuals = (sc[sel] - sc["Actual"]).to_numpy()

        c_err1, c_err2 = st.columns(2)
        with c_err1:
            st.plotly_chart(charts.residual_hist(residuals), use_container_width=True, key="perf_res")
        with c_err2:
            st.plotly_chart(charts.error_by_hour_bar(T["by_hour"]), use_container_width=True, key="perf_by_hour")

        section("Feature Importance (Permutation on Validation)")
        st.plotly_chart(charts.importance_bar(T["importance"]), use_container_width=True, key="perf_importance")


# ----------------------------------------------------------------------------- PAGE 5: Data Quality
def page_quality(client, meta, T, model_stamp):
    q = T["quality"]
    hourly = T["hourly"]

    hero(
        "Data Quality & Preprocessing Assurance",
        "Transparent verification of data integrity: unit conversion, incomplete hour exclusion, "
        "structural zero handling, and European Daylight Saving Time (DST) policy.",
        [("Status: Verified Clean", "emerald"), ("Policy: Documented & Non-Leaking", "blue")],
    )

    kpi_row([
        ("Raw 15-Min Records", f"{q.get('raw_rows', 0):,}", "rows", f"{q.get('raw_start', '')[:10]} → {q.get('raw_end', '')[:10]}", "blue"),
        ("Valid Hourly kWh Rows", f"{q.get('hourly_rows', 0):,}", "hours", "4 x 0.25 kW conversion", "emerald"),
        ("Structural Zeros Trimmed", f"{q.get('structural_zero_rows_trimmed', 0):,}", "rows", "pre-activation zeros", "amber" if q.get("structural_zero_rows_trimmed", 0) else "emerald"),
        ("DST Flagged Hours", str(q.get("dst_flagged_hours", 0)), "hours", "excluded from scoring", "amber"),
        ("Incomplete Hours Excluded", str(q.get("incomplete_hours", 0)), "hours", "< 4 valid readings", "rose" if q.get("incomplete_hours", 0) else "emerald"),
    ])

    section("Automated Integrity Verification Checks")
    # Dynamic data quality verification computed directly on loaded hourly records
    is_monotonic = bool(hourly.index.is_monotonic_increasing)
    dup_ts = int(hourly.index.duplicated().sum())
    missing_vals = int(hourly["kwh"].isna().sum())
    inf_vals = int(np.isinf(hourly["kwh"]).sum())
    negative_vals = int((hourly["kwh"] < 0).sum())
    incomplete_hrs = int(hourly["incomplete"].sum()) if "incomplete" in hourly.columns else int(q.get("incomplete_hours", 0))
    dst_hrs = int(hourly["dst"].sum()) if "dst" in hourly.columns else int(q.get("dst_flagged_hours", 0))

    checks = [
        ("Timestamp Ordering & Continuity", f"Monotonic: {is_monotonic} | Duplicate Timestamps: {dup_ts}", "✅ Pass" if is_monotonic and dup_ts == 0 else "⚠️ Warning"),
        ("Numeric Values & Non-Finite Checks", f"{missing_vals} NaN | {inf_vals} Inf / -Inf values", "✅ Pass" if missing_vals == 0 and inf_vals == 0 else "⚠️ Handled"),
        ("Negative Consumption Bounds", f"{negative_vals} negative values detected in hourly series", "✅ Pass" if negative_vals == 0 else "⚠️ Handled"),
        ("15-Minute Grid Cadence", f"{q.get('off_grid_intervals', 0)} off-grid raw intervals", "✅ Pass" if q.get("off_grid_intervals", 0) == 0 else "⚠️ Handled"),
        ("Structural Zeros (Client Entry Date)", f"{q.get('structural_zero_rows_trimmed', 0):,} inactive prefix rows trimmed prior to activation", "ℹ️ Handled"),
        ("Incomplete Hours (< 4 Readings)", f"{incomplete_hrs} incomplete hours detected & excluded from scoring", "✅ Pass" if incomplete_hrs == 0 else "ℹ️ Excluded"),
        ("Daylight Saving Time (DST)", f"{dst_hrs} hours flagged; policy: {q.get('dst_policy', 'flag_only')}", "ℹ️ Flagged"),
        ("Autoregressive Lag Warm-Up Window", "168 initial hours (7 days) reserved for lag-168 & roll-168 feature priming", "ℹ️ Reserved"),
        ("External Meteorological / Weather Data", "Not recorded (smart meter consumption-only dataset)", "— Not Recorded"),
        ("Telemetry Latency & Grid Frequency", "Not recorded (smart meter consumption-only dataset)", "— Not Recorded"),
    ]
    st.dataframe(pd.DataFrame(checks, columns=["Verification Check", "Result", "Status"]), hide_index=True)

    section("Flagged Hours Inspection")
    flagged = hourly[(hourly["dst"]) | (hourly["imputed"]) | (hourly["incomplete"])].copy()
    if not flagged.empty:
        flagged["reason"] = [
            ", ".join(n for n, f in (("Daylight-Saving", d), ("Incomplete Interval", i), ("Imputed", m)) if f)
            for d, i, m in zip(flagged["dst"], flagged["incomplete"], flagged["imputed"])
        ]
        st.dataframe(flagged[["kwh", "reason"]].round(2).reset_index().rename(columns={"timestamp": "Timestamp", "kwh": "Energy (kWh)", "reason": "Reason"}), hide_index=True)
    else:
        st.info("No anomalous or flagged hours in this series.")

    c1, c2, c3 = st.columns(3)
    with c1:
        card(
            "Measurement Unit Convention",
            "<p>The source dataset provides average power in <strong>kilowatts (kW)</strong> at 15-minute intervals. "
            "Hourly energy consumption is calculated as:<br><code>E_h = sum_{j=1}^4 P_{h,j} * 0.25 kWh</code>.<br>"
            "Timestamps consistently represent the <strong>end</strong> of the hourly observation period.</p>",
        )
    with c2:
        card(
            "Autoregressive Feature Warm-Up",
            "<p>Because models use up to 168-hour (7-day) autoregressive lags and rolling windows, "
            "the first 168 hours following the client entry date serve as a <strong>feature warm-up window</strong>. "
            "Model training begins only once full historical lag context is available.</p>",
        )
    with c3:
        card(
            "Dataset Fingerprint & Provenance",
            f"<p><strong>SHA-256 Digest:</strong> <code>{meta['dataset']['sha256'][:24]}...</code></p>"
            f"<p><strong>Source Note:</strong> {meta['dataset']['note']}</p>",
        )


# ----------------------------------------------------------------------------- PAGE 6: About Project
def page_about(client, meta, T, model_stamp):
    hero(
        "About GridSense",
        "An AI-based electricity demand prediction and visualization system engineered for transparent, "
        "scientifically validated smart grid demand intelligence.",
        [("Architecture: Modular Python", "blue"), ("Frontend: Streamlit SaaS", "indigo")],
    )

    c1, c2 = st.columns(2)
    with c1:
        card(
            "1. Project Mission & Objectives",
            "<p>GridSense addresses the critical challenge of predicting electrical load in modern smart power grids. "
            "Accurate short-term forecasting is essential for grid operators to balance supply and demand, optimize generator dispatch, "
            "minimize operational costs, and facilitate the smooth integration of intermittent renewable energy sources.</p>",
        )
        card(
            "2. Leakage-Free Feature Engineering",
            "<p>Every feature used to predict hour <em>t</em> strictly uses data available up to <em>t − 1</em>. "
            "Features include calendar signals (hour, day of week, day of month, month, weekend indicator), "
            "autoregressive lags (1 hour, 24 hours, 168 hours), and shifted rolling statistics (24h mean, 24h std, 168h mean). "
            "Target observations never enter their own features.</p>",
        )
        card(
            "3. The Five Forecasting Models",
            "<li><strong>Persistence:</strong> Assumes consumption equals the previous hour (t-1).</li>"
            "<li><strong>Daily Seasonal Naive:</strong> Assumes consumption equals the same hour yesterday (t-24).</li>"
            "<li><strong>Weekly Seasonal Naive:</strong> Assumes consumption equals the same hour last week (t-168), capturing 7-day weekly recurrence patterns.</li>"
            "<li><strong>Random Forest:</strong> An ensemble of decision trees capturing non-linear interactions.</li>"
            "<li><strong>XGBoost:</strong> Gradient boosted decision trees optimized for time-series tabular data.</li>"
            "</ul>",
        )

    with c2:
        card(
            "4. Chronological Validation Pipeline",
            "<p>Rather than using random train/test splits that leak temporal structure, GridSense uses a strict "
            "<strong>70% Train / 15% Validation / 15% Test</strong> chronological partition. "
            "The final test set remains untouched during model selection and is evaluated only once to provide unbiased performance estimates.</p>",
        )
        card(
            "5. Statistical High-Demand Alerts",
            "<p>Thresholds are derived strictly from the training partition's 95th percentile (P95). "
            "Predicted consumption exceeding this threshold generates a high-demand flag to notify operators of unusual peak usage.</p>",
        )
        if IS_SYNTH:
            card(
                "6. Dataset Provenance & Academic Disclaimers",
                "<ul>"
                "<li><strong>Dataset Origin:</strong> Bundled with a <em>Historical Synthetic Electricity Consumption Dataset</em> (generated via <code>scripts/generate_demo_dataset.py</code> with seed 2024). Values are simulated benchmark signals, not actual customer smart-meter measurements.</li>"
                "<li><strong>Scope:</strong> GridSense is an academic forecasting prototype evaluated using synthetic historical electricity consumption data. It has not been validated against live electricity-grid measurements or deployed in a real smart-grid environment.</li>"
                "<li>High-demand alerts are <strong>statistical anomaly flags</strong> relative to training percentiles, not physical transformer or grid failure detections.</li>"
                "</ul>",
            )
        else:
            card(
                "6. Dataset Provenance & Academic Disclaimers",
                "<ul>"
                "<li><strong>Official Dataset:</strong> Genuine <em>UCI ElectricityLoadDiagrams20112014</em> (UCI ID: 321, DOI: <a href='https://doi.org/10.24432/C58C86' target='_blank'>10.24432/C58C86</a>), donated by Artur Trindade (INESC Porto / University of Porto, 2015) under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.</li>"
                "<li><strong>Scope & Context:</strong> Historical 15-minute power demand (kW) from 370 client metering series in Portugal across 2011–2014. Converted to hourly energy in kilowatt-hours (kWh) via $E_h = \\sum_{j=1}^4 P_{h,j} \\times 0.25$.</li>"
                "<li><strong>Academic Disclaimer:</strong> GridSense is an academic forecasting prototype evaluated using historical Portuguese smart-meter telemetry. It has not been validated against live Indian electricity-grid measurements or deployed in a real-time SCADA system.</li>"
                "<li><strong>Aggregate Terminology:</strong> The portfolio series represents the <em>Aggregate Consumption of 370 Observed Clients</em>, not total Portuguese national grid consumption.</li>"
                "<li>High-demand alerts are <strong>statistical anomaly flags</strong> ($P_{95}$ relative to training data), not physical thermal or transformer limit detections.</li>"
                "</ul>",
            )


PAGE_FUNCS = {
    "Overview": page_overview,
    "Historical Analysis": page_historical,
    "Forecasting": page_forecasting,
    "Model Performance": page_performance,
    "Data Quality": page_quality,
    "About Project": page_about,
}


# ----------------------------------------------------------------------------- Main Application Flow
def main() -> None:
    default_client = CFG.get("dataset", {}).get("default_client", "MT_001")
    if not DATA_AVAILABLE and not artefacts_exist(model_dir(default_client)):
        hero(
            "Dataset Not Found",
            "Expected dataset file at 'data/raw/real/LD2011_2014.txt'. "
            "To download and verify the official UCI dataset, run: python scripts/download_real_data.py",
            [("Setup Required", "rose")],
        )
        st.stop()

    with st.sidebar:
        brand()
        clients = get_clients(DATA_PATH_STR, DATA_MTIME)
        default_client = CFG.get("dataset", {}).get("default_client", "MT_001")
        client = st.selectbox(
            "Select Electricity Series / Client",
            clients,
            index=clients.index(default_client) if default_client in clients else 0,
            key="selected_client",
            format_func=lambda c: ("● " if artefacts_exist(model_dir(c)) else "○ ") + c,
            help="● Trained model exists · ○ Model not trained yet. PORTFOLIO = Aggregate of clients.",
        )

        st.write("")
        st.markdown("**Navigation**")
        page = st.radio(
            "Navigation",
            PAGES,
            format_func=lambda p: f"{ICONS[p]}  {p}",
            label_visibility="collapsed",
            key="main_nav",
        )
        st.markdown("---")

    # If the selected series has not been trained yet
    if not artefacts_exist(model_dir(client)):
        with st.sidebar:
            st.markdown('<span class="gs-pill amber">● Model Not Trained</span>', unsafe_allow_html=True)
        train_view(client)
        disclaimer()
        return

    stamp = _stamp(client)
    try:
        _, meta = get_model(client, stamp)
        T = get_tables(client, stamp)
    except (ForecastError, FileNotFoundError) as exc:
        st.error(f"Could not load trained artefacts for {client}: {exc}")
        if st.button("↻ Retrain This Series", key="btn_retrain_err"):
            for p in (model_dir(client) / "metadata.json",):
                p.unlink(missing_ok=True)
            st.rerun()
        return

    with st.sidebar:
        st.markdown(
            f'<div style="margin-bottom:0.75rem;">'
            f'<span class="gs-pill emerald">● Model Ready</span>'
            f'<span class="gs-pill indigo">{meta.get("selected_model", "Selected")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Trained: {meta.get('created_utc', '')[:16].replace('T', ' ')} UTC · {meta.get('training_seconds', 0):.0f}s")
        if IS_SYNTH:
            st.markdown('<span class="gs-pill amber">Synthetic Demo Data</span>', unsafe_allow_html=True)

        if st.button("↻ Retrain This Series", key="btn_sidebar_retrain"):
            bar = st.progress(0.0, text="Retraining model pipeline...")
            run_experiment(client, CFG, progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    # Render selected page
    PAGE_FUNCS[page](client, meta, T, stamp)
    disclaimer()


if __name__ == "__main__":
    main()
