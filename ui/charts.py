"""Plotly figure builders (clean, modern SaaS light theme, accessible palette) (Phases 5 & 7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BLUE = "#2563EB"
INDIGO = "#6366F1"
EMERALD = "#10B981"
AMBER = "#F59E0B"
ROSE = "#EF4444"
SLATE = "#64748B"
DARK = "#0F172A"
GRID = "#F1F5F9"
BORDER = "#E2E8F0"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _style(fig: go.Figure, height: int = 400, title: str | None = None, unified: bool = True) -> go.Figure:
    layout_opts = dict(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=12, r=12, t=48 if title else 20, b=12),
        font=dict(family="Inter, sans-serif", color="#334155", size=12),
        legend=dict(
            orientation="h",
            y=1.12,
            x=0.0,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(size=11, color="#475569"),
        ),
        hovermode="closest",
    )
    if title:
        layout_opts["title"] = dict(
            text=title,
            x=0.01,
            y=0.96,
            font=dict(size=14, color="#0F172A", family="Inter, sans-serif"),
        )
    fig.update_layout(**layout_opts)
    fig.update_xaxes(
        gridcolor=GRID,
        linecolor=BORDER,
        zeroline=False,
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#0F172A", size=12),
    )
    fig.update_yaxes(
        gridcolor=GRID,
        linecolor=BORDER,
        zeroline=False,
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#0F172A", size=12),
    )
    return fig


def forecast_fan(
    history: pd.Series,
    fc: pd.DataFrame,
    actual: pd.Series | None = None,
    threshold: float | None = None,
    show_interval: bool = True,
    scenario: pd.Series | None = None,
    height: int = 440,
) -> go.Figure:
    """Plot historical observations, forecast path, prediction interval, and high-demand alerts."""
    fig = go.Figure()
    # Historical observed
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            name="Observed History",
            mode="lines",
            line=dict(color="#475569", width=1.8),
        )
    )

    # Prediction intervals
    if show_interval and {"lower_kwh", "upper_kwh"} <= set(fc.columns):
        fig.add_trace(
            go.Scatter(
                x=fc["target_time"],
                y=fc["upper_kwh"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc["target_time"],
                y=fc["lower_kwh"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(99, 102, 241, 0.14)",
                name="Prediction Interval (Empirical)",
                hoverinfo="skip",
            )
        )

    # Forecast path
    fig.add_trace(
        go.Scatter(
            x=fc["target_time"],
            y=fc["predicted_kwh"],
            name="Forecast (kWh)",
            mode="lines+markers",
            line=dict(color=BLUE, width=2.6),
            marker=dict(size=5, color=BLUE),
        )
    )

    if scenario is not None:
        fig.add_trace(
            go.Scatter(
                x=scenario.index,
                y=scenario.values,
                name="Peak-shaving Scenario",
                mode="lines",
                line=dict(color=EMERALD, width=2, dash="dash"),
            )
        )

    if actual is not None and len(actual):
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual.values,
                name="Actual (Replay)",
                mode="lines",
                line=dict(color="#8B5CF6", width=2.2, dash="dot"),
            )
        )

    # Highlight High-Demand alert points
    if "alert" in fc.columns and fc["alert"].any():
        a = fc[fc["alert"]]
        fig.add_trace(
            go.Scatter(
                x=a["target_time"],
                y=a["predicted_kwh"],
                mode="markers",
                name="High-Demand Alert",
                marker=dict(
                    color=ROSE,
                    size=11,
                    symbol="diamond",
                    line=dict(color="#FFFFFF", width=1.5),
                ),
            )
        )

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color=AMBER,
            annotation_text=f"High Threshold: {threshold:,.1f} kWh",
            annotation_font_color=AMBER,
            annotation_position="top right",
        )

    # Vertical line demarcating forecast origin
    if len(fc):
        origin_t = fc["target_time"].iloc[0] - pd.Timedelta(minutes=30)
        fig.add_shape(
            type="line",
            x0=origin_t,
            x1=origin_t,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#CBD5E1", dash="dash", width=1.5),
        )

    fig.update_yaxes(title="Energy Consumption (kWh)")
    fig.update_xaxes(title="Timestamp")
    return _style(fig, height)


def history_chart(s: pd.Series, threshold: float | None = None, label: str = "Electricity Consumption (kWh)") -> go.Figure:
    """Historical consumption trend line with interactive range slider."""
    fig = go.Figure(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name=label,
            line=dict(color=BLUE, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.05)",
            hovertemplate="%{x}<br>Energy: %{y:.2f} kWh<extra></extra>",
        )
    )
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color=AMBER,
            annotation_text=f"P95 Threshold: {threshold:.1f} kWh",
            annotation_font_color=AMBER,
            annotation_position="top right",
        )
    fig.update_xaxes(rangeslider_visible=True, title="Date / Time")
    fig.update_yaxes(title=label)
    return _style(fig, 420)


def heatmap_hour_weekday(s: pd.Series, title: str = "Average Consumption by Hour and Weekday (kWh)") -> go.Figure:
    """Hour-of-day vs day-of-week intensity heatmap."""
    df = pd.DataFrame({"v": s.values, "h": s.index.hour, "d": s.index.dayofweek})
    pv = df.pivot_table(index="d", columns="h", values="v", aggfunc="mean").reindex(
        index=range(7), columns=range(24)
    )
    fig = go.Figure(
        go.Heatmap(
            z=pv.values,
            x=list(range(24)),
            y=DAYS,
            colorscale=[[0, "#EFF6FF"], [0.5, "#60A5FA"], [1, "#1D4ED8"]],
            colorbar=dict(title="kWh", thickness=14, len=0.8),
            hovertemplate="%{y} at %{x}:00 → %{z:.2f} kWh<extra></extra>",
        )
    )
    fig.update_xaxes(title="Hour of Day (Interval End)", dtick=2)
    fig.update_yaxes(autorange="reversed")
    return _style(fig, 320, title, unified=False)


def hourly_profile(s: pd.Series) -> go.Figure:
    """Diurnal load shape comparing weekdays vs weekends."""
    wk = s[s.index.dayofweek < 5].groupby(s[s.index.dayofweek < 5].index.hour).mean()
    we = s[s.index.dayofweek >= 5].groupby(s[s.index.dayofweek >= 5].index.hour).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wk.index, y=wk.values, name="Weekdays", line=dict(color=BLUE, width=2.8)))
    fig.add_trace(go.Scatter(x=we.index, y=we.values, name="Weekends", line=dict(color=INDIGO, width=2.4, dash="dash")))
    fig.update_xaxes(title="Hour of Day", dtick=2)
    fig.update_yaxes(title="Average Energy (kWh)")
    return _style(fig, 320, "Daily Diurnal Profile (Weekdays vs Weekends)")


def monthly_bars(s: pd.Series) -> go.Figure:
    """Monthly average consumption seasonality bars."""
    m = s.groupby(s.index.month).mean()
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    labels = [names[i - 1] for i in m.index]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=m.values,
            marker=dict(color=BLUE, opacity=0.85),
            hovertemplate="%{x}: %{y:.2f} kWh<extra></extra>",
        )
    )
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Mean Hourly kWh")
    return _style(fig, 320, "Monthly Seasonality (Mean Hourly kWh)", unified=False)


def weekday_bars(s: pd.Series) -> go.Figure:
    """Weekday average consumption comparison bars."""
    d = s.groupby(s.index.dayofweek).mean().reindex(range(7))
    fig = go.Figure(
        go.Bar(
            x=DAYS,
            y=d.values,
            marker=dict(color=[BLUE if i < 5 else INDIGO for i in range(7)], opacity=0.85),
            hovertemplate="%{x}: %{y:.2f} kWh<extra></extra>",
        )
    )
    fig.update_xaxes(title="Day of Week")
    fig.update_yaxes(title="Mean Hourly kWh")
    return _style(fig, 320, "Average Energy by Day of Week", unified=False)


def distribution(s: pd.Series, p25: float | None = None, p95: float | None = None) -> go.Figure:
    """Histogram of hourly consumption showing threshold percentiles."""
    fig = go.Figure(
        go.Histogram(
            x=s.values,
            nbinsx=50,
            marker=dict(color=BLUE, line=dict(color="#FFFFFF", width=0.5)),
            opacity=0.8,
            hovertemplate="Energy: %{x:.1f} kWh<br>Count: %{y}<extra></extra>",
        )
    )
    if p25 is not None:
        fig.add_vline(
            x=p25,
            line_dash="dash",
            line_color=EMERALD,
            annotation_text=f"P25 ({p25:.1f} kWh)",
            annotation_font_color=EMERALD,
            annotation_position="top left",
        )
    if p95 is not None:
        fig.add_vline(
            x=p95,
            line_dash="dot",
            line_color=ROSE,
            annotation_text=f"P95 ({p95:.1f} kWh)",
            annotation_font_color=ROSE,
            annotation_position="top right",
        )
    fig.update_xaxes(title="Hourly Energy Consumption (kWh)")
    fig.update_yaxes(title="Number of Hours")
    return _style(fig, 320, "Consumption Distribution and Demand Thresholds", unified=False)


def leaderboard_mae_chart(metrics: pd.DataFrame) -> go.Figure:
    """Bar chart comparing MAE across all 5 models."""
    colors = [INDIGO if k == "ml" else SLATE for k in metrics["kind"]]
    fig = go.Figure(
        go.Bar(
            x=metrics["model_name"],
            y=metrics["MAE"],
            marker=dict(color=colors),
            text=[f"{v:.3f} kWh" for v in metrics["MAE"]],
            textposition="auto",
            hovertemplate="%{x}: MAE = %{y:.3f} kWh<extra></extra>",
        )
    )
    fig.update_yaxes(title="Mean Absolute Error (kWh) — Lower is Better")
    fig.update_xaxes(title="Model")
    return _style(fig, 340, "Model Comparison: MAE (Test Period)", unified=False)


def leaderboard_rmse_chart(metrics: pd.DataFrame) -> go.Figure:
    """Bar chart comparing RMSE across all 5 models."""
    colors = [INDIGO if k == "ml" else SLATE for k in metrics["kind"]]
    fig = go.Figure(
        go.Bar(
            x=metrics["model_name"],
            y=metrics["RMSE"],
            marker=dict(color=colors, opacity=0.85),
            text=[f"{v:.3f} kWh" for v in metrics["RMSE"]],
            textposition="auto",
            hovertemplate="%{x}: RMSE = %{y:.3f} kWh<extra></extra>",
        )
    )
    fig.update_yaxes(title="Root Mean Squared Error (kWh) — Lower is Better")
    fig.update_xaxes(title="Model")
    return _style(fig, 340, "Model Comparison: RMSE (Test Period)", unified=False)


def actual_vs_pred(preds: pd.DataFrame, model_col: str, baseline_col: str | None = None) -> go.Figure:
    """Interactive actual vs predicted comparison over test period."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=preds.index,
            y=preds["Actual"],
            name="Actual (kWh)",
            line=dict(color=DARK, width=1.8),
        )
    )
    if baseline_col and baseline_col in preds.columns:
        fig.add_trace(
            go.Scatter(
                x=preds.index,
                y=preds[baseline_col],
                name=f"Baseline ({baseline_col})",
                line=dict(color=SLATE, width=1.3, dash="dot"),
            )
        )
    if model_col in preds.columns:
        fig.add_trace(
            go.Scatter(
                x=preds.index,
                y=preds[model_col],
                name=f"ML Model ({model_col})",
                line=dict(color=BLUE, width=2.4),
            )
        )
    fig.update_yaxes(title="Electricity Consumption (kWh)")
    fig.update_xaxes(rangeslider_visible=True, title="Test Period Timestamp")
    return _style(fig, 420, "Actual vs. Predicted Consumption (Untouched Test Period)")


def residual_hist(residuals: np.ndarray) -> go.Figure:
    """Residual error histogram (predicted - actual)."""
    fig = go.Figure(
        go.Histogram(
            x=residuals,
            nbinsx=50,
            marker=dict(color=INDIGO, line=dict(color="#FFFFFF", width=0.5)),
            opacity=0.85,
            hovertemplate="Error: %{x:.2f} kWh<br>Count: %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=ROSE, line_dash="dash", annotation_text="Zero Error")
    fig.update_xaxes(title="Prediction Error = Predicted − Actual (kWh)")
    fig.update_yaxes(title="Frequency")
    return _style(fig, 300, "Prediction Error Distribution", unified=False)


def error_by_hour_bar(df: pd.DataFrame) -> go.Figure:
    """MAE broken down by hour of the day."""
    fig = go.Figure(
        go.Bar(
            x=df["hour"],
            y=df["MAE_kWh"],
            marker=dict(color=df["MAE_kWh"], colorscale=[[0, "#93C5FD"], [1, BLUE]]),
            hovertemplate="Hour %{x}:00: MAE = %{y:.3f} kWh<extra></extra>",
        )
    )
    fig.update_xaxes(title="Hour of Day", dtick=2)
    fig.update_yaxes(title="MAE (kWh)")
    return _style(fig, 300, "Prediction Error by Hour of Day", unified=False)


def importance_bar(df: pd.DataFrame) -> go.Figure:
    """Permutation feature importance bar chart."""
    d = df.sort_values("importance_mae_kwh")
    fig = go.Figure(
        go.Bar(
            x=d["importance_mae_kwh"],
            y=d["feature"],
            orientation="h",
            marker=dict(color=BLUE, opacity=0.9),
            hovertemplate="%{y}: +%{x:.3f} kWh error when shuffled<extra></extra>",
        )
    )
    fig.update_xaxes(title="Increase in MAE when Shuffled (kWh)")
    fig.update_yaxes(title="Feature")
    return _style(fig, 350, "Feature Importance (Permutation on Validation)", unified=False)


def horizon_error(mae_by_h: list[float]) -> go.Figure:
    """Error progression across multi-step forecast horizons."""
    h = list(range(1, len(mae_by_h) + 1))
    fig = go.Figure(
        go.Scatter(
            x=h,
            y=mae_by_h,
            mode="lines+markers",
            line=dict(color=AMBER, width=2.5),
            marker=dict(size=6, color=AMBER),
            hovertemplate="Horizon +%{x}h: MAE = %{y:.3f} kWh<extra></extra>",
        )
    )
    fig.update_xaxes(title="Forecast Horizon Ahead (Hours)")
    fig.update_yaxes(title="Backtest MAE (kWh)")
    return _style(fig, 300, "Forecast Accuracy Decay with Horizon", unified=False)
