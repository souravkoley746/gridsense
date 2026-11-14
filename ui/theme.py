"""Modern SaaS Light Theme, visual identity, cards, and KPI tiles (Phase 5).

Palette:
- Background: #F8FAFC
- Cards: #FFFFFF
- Text: Primary #0F172A, Secondary #334155, Muted #64748B
- Borders: #E2E8F0
- Primary Blue: #2563EB
- Emerald (Low / Pass): #10B981
- Amber (Normal / Warning): #F59E0B
- Rose (High Alert / Critical): #EF4444
- Indigo (Forecast Accent): #6366F1
"""
from __future__ import annotations

import html
import streamlit as st

BLUE = "#2563EB"
EMERALD = "#10B981"
AMBER = "#F59E0B"
ROSE = "#EF4444"
INDIGO = "#6366F1"
TEXT_PRIMARY = "#0F172A"
TEXT_SECONDARY = "#334155"
TEXT_MUTED = "#64748B"
BORDER = "#E2E8F0"
CARD_BG = "#FFFFFF"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0F172A;
}

.stApp {
    background-color: #F8FAFC;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3.5rem;
    max-width: 1440px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
    box-shadow: 1px 0 3px rgba(0, 0, 0, 0.02);
}

/* Brand header in sidebar */
.gs-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0 1.25rem 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 1.25rem;
}

.gs-logo {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: grid;
    place-items: center;
    font-size: 1.35rem;
    background: linear-gradient(135deg, #2563EB, #6366F1);
    color: #FFFFFF;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
}

.gs-brand h1 {
    font-size: 1.25rem;
    margin: 0;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #0F172A;
}

.gs-brand small {
    color: #64748B;
    display: block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Modern Hero Banner */
.gs-hero {
    padding: 1.5rem 1.75rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.gs-hero h2 {
    margin: 0 0 0.35rem 0;
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #0F172A;
}

.gs-hero p {
    margin: 0;
    color: #475569;
    font-size: 0.95rem;
    line-height: 1.5;
    max-width: 65rem;
}

/* Badges and Pills */
.gs-pill {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid #E2E8F0;
    background: #F1F5F9;
    color: #334155;
    margin-right: 0.4rem;
}

.gs-pill.ok, .gs-pill.emerald, .gs-pill.low {
    color: #065F46;
    border-color: #A7F3D0;
    background: #ECFDF5;
}

.gs-pill.warn, .gs-pill.amber, .gs-pill.normal {
    color: #92400E;
    border-color: #FDE68A;
    background: #FFFBEB;
}

.gs-pill.bad, .gs-pill.rose, .gs-pill.high {
    color: #991B1B;
    border-color: #FECACA;
    background: #FEF2F2;
}

.gs-pill.indigo, .gs-pill.blue {
    color: #1E40AF;
    border-color: #BFDBFE;
    background: #EFF6FF;
}

/* KPI Cards */
.gs-kpi {
    padding: 0.85rem 0.65rem;
    border-radius: 12px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.gs-kpi .l {
    color: #64748B;
    font-size: 0.70rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.gs-kpi .v {
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #0F172A;
    margin: 0.15rem 0;
    word-break: break-word;
}

.gs-kpi .v span {
    font-size: 0.75rem;
    color: #64748B;
    font-weight: 600;
    margin-left: 0.2rem;
}

.gs-kpi .s {
    color: #64748B;
    font-size: 0.8rem;
    margin-top: 0.25rem;
}

.gs-kpi.blue { border-top: 3px solid #2563EB; }
.gs-kpi.emerald { border-top: 3px solid #10B981; }
.gs-kpi.amber { border-top: 3px solid #F59E0B; }
.gs-kpi.rose { border-top: 3px solid #EF4444; }
.gs-kpi.indigo { border-top: 3px solid #6366F1; }
.gs-kpi.slate { border-top: 3px solid #64748B; }

/* Content Cards */
.gs-card {
    padding: 1.25rem 1.4rem;
    border-radius: 12px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    margin-bottom: 1rem;
}

.gs-card h4 {
    margin: 0 0 0.5rem 0;
    font-size: 1.05rem;
    font-weight: 700;
    color: #0F172A;
}

.gs-card p, .gs-card li {
    color: #334155;
    font-size: 0.92rem;
    line-height: 1.55;
}

/* Prediction Highlight Card */
.gs-prediction-card {
    padding: 1.75rem 2rem;
    border-radius: 16px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    margin: 1.25rem 0;
}

.gs-prediction-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1.25rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #F1F5F9;
}

/* Demand Status Banners */
.gs-status-banner {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 0.9rem 1.25rem;
    border-radius: 10px;
    font-size: 0.92rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

.gs-status-banner.low {
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    color: #065F46;
}

.gs-status-banner.normal {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    color: #92400E;
}

.gs-status-banner.high {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    color: #991B1B;
}

/* Section titles */
.gs-section {
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748B;
    font-weight: 700;
    margin: 1.5rem 0 0.65rem 0;
}

/* Disclaimer footer */
.gs-disclaimer {
    color: #64748B;
    font-size: 0.8rem;
    line-height: 1.5;
    border-top: 1px solid #E2E8F0;
    margin-top: 2.5rem;
    padding-top: 1rem;
}

div[data-testid="stTabs"] button {
    font-weight: 600;
    color: #475569;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563EB;
}

div[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    overflow: hidden;
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def brand() -> None:
    st.markdown(
        '<div class="gs-brand">'
        '<div class="gs-logo">⚡</div>'
        '<div><h1>GridSense</h1><small>Smart Grid Demand Intelligence</small></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, pills: list[tuple[str, str]] | None = None) -> None:
    pill_html = "".join(f'<span class="gs-pill {html.escape(k)}">{html.escape(t)}</span>' for t, k in (pills or []))
    st.markdown(
        f'<div class="gs-hero">'
        f'<div style="margin-bottom:.65rem">{pill_html}</div>'
        f'<h2>{html.escape(title)}</h2>'
        f'<p>{html.escape(subtitle)}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, unit: str = "", sub: str = "", tone: str = "blue") -> str:
    u = f"<span>{html.escape(unit)}</span>" if unit else ""
    s_div = f'<div class="s">{html.escape(sub)}</div>' if sub else ""
    return (
        f'<div class="gs-kpi {tone}">'
        f'<div><div class="l">{html.escape(label)}</div>'
        f'<div class="v">{html.escape(value)}{u}</div></div>'
        f'{s_div}</div>'
    )


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(kpi(*it), unsafe_allow_html=True)


def section(text: str) -> None:
    st.markdown(f'<div class="gs-section">{html.escape(text)}</div>', unsafe_allow_html=True)


def card(title: str, body_html: str) -> None:
    st.markdown(
        f'<div class="gs-card"><h4>{html.escape(title)}</h4>{body_html}</div>',
        unsafe_allow_html=True,
    )


def demand_status_badge(level: str, text: str) -> None:
    lvl = level.lower()
    icon = "✓" if lvl == "low" else ("ℹ" if lvl == "normal" else "⚠")
    st.markdown(
        f'<div class="gs-status-banner {lvl}">'
        f'<span><strong>{icon} Demand Status: {html.escape(level)}</strong> — {html.escape(text)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    st.markdown(
        '<div class="gs-disclaimer">'
        '<strong>Project Boundary & Academic Disclaimer:</strong> GridSense is a decision-support and forecasting prototype '
        'trained on historical electricity consumption data (2011–2014). It is not a real-time smart grid control system, '
        'nor does it interface with physical grid equipment. A “High-Demand Alert” is a statistical flag indicating consumption '
        'exceeding historical baseline thresholds (e.g., 95th percentile); it does not signify transformer overload, feeder failure, '
        'or an impending blackout.'
        '</div>',
        unsafe_allow_html=True,
    )
