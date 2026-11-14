"""Comprehensive, high-resolution screenshot capture for GridSense demonstration.

Captures:
1. Full-page screenshots of all 6 pages (with complete scrolling and chart rendering).
2. Dedicated high-focus captures of Hero Sections and Core Features:
   - Overview: Hero Card, KPI Metrics Grid, Interactive Timeline, Diurnal Profiles.
   - Historical: Date Filter, Diurnal/Calendar Tab, Peak Load Tab.
   - Forecasting: Hero Controls, Target KPI, 24h Fan Chart with Prediction Intervals, Multi-Step Forecast Table.
   - Performance: 5-Model Leaderboard, MAE/RMSE Comparison Bar Charts, Actual vs Predicted, Feature Importance.
   - Quality: Integrity Verification Checks Table, Flagged Hours Inspection.
   - About: Mission, Methodology, Provenance & Boundaries.
"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS_DIR = ROOT / "screenshots"
ARTIFACT_DIR = Path(r"C:\Users\sourav\.gemini\antigravity\brain\a2f65c3b-8d2c-45a4-b7b5-6900fae8d669")

async def scroll_and_settle(page):
    """Scroll down and up to trigger lazy rendering of Plotly charts and tables."""
    await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
    await page.wait_for_timeout(2500)
    await page.evaluate("() => window.scrollTo(0, 0)")
    await page.wait_for_timeout(1500)

async def save_screenshot(page, filename, full_page=True, locator=None):
    out_root = SCREENSHOTS_DIR / filename
    out_art = ARTIFACT_DIR / filename
    if locator:
        await locator.screenshot(path=str(out_root))
        await locator.screenshot(path=str(out_art))
    else:
        await page.screenshot(path=str(out_root), full_page=full_page)
        await page.screenshot(path=str(out_art), full_page=full_page)
    print(f"  [Captured] {filename}")

async def run_capture():
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # 1440x900 desktop viewport for clean layout rendering
        context = await browser.new_context(
            viewport={"width": 1440, "height": 960},
            device_scale_factor=2.0  # Crisp Retina 2x resolution
        )
        page = await context.new_page()

        print("Navigating to GridSense at http://127.0.0.1:8501...")
        await page.goto("http://127.0.0.1:8501", wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(3000)

        # -------------------------------------------------------------
        # PAGE 1: OVERVIEW
        # -------------------------------------------------------------
        print("\nCapturing Page 1: Overview...")
        await scroll_and_settle(page)
        await save_screenshot(page, "page_1_overview_full.png", full_page=True)
        await save_screenshot(page, "page_1_overview.png", full_page=True)
        
        # Hero section focus
        hero1 = page.locator(".gs-hero-card").first
        if await hero1.count() > 0:
            await save_screenshot(page, "feature_overview_hero.png", locator=hero1)

        # -------------------------------------------------------------
        # PAGE 2: HISTORICAL ANALYSIS
        # -------------------------------------------------------------
        print("\nCapturing Page 2: Historical Analysis...")
        await page.get_by_text("Historical Analysis", exact=False).first.click()
        await page.wait_for_timeout(3000)
        await scroll_and_settle(page)
        await save_screenshot(page, "page_2_historical_full.png", full_page=True)
        await save_screenshot(page, "page_2_historical.png", full_page=True)

        # Diurnal & Calendar Patterns Tab
        diurnal_tab = page.get_by_role("tab", name="Diurnal & Calendar Patterns").first
        if await diurnal_tab.count() > 0:
            await diurnal_tab.click()
            await page.wait_for_timeout(2000)
            await save_screenshot(page, "feature_historical_diurnal_patterns.png", full_page=False)

        # Peak Periods & Distribution Tab
        peak_tab = page.get_by_role("tab", name="Peak Periods & Distribution").first
        if await peak_tab.count() > 0:
            await peak_tab.click()
            await page.wait_for_timeout(2000)
            await save_screenshot(page, "feature_historical_peak_distribution.png", full_page=False)

        # -------------------------------------------------------------
        # PAGE 3: FORECASTING (CORE FEATURE)
        # -------------------------------------------------------------
        print("\nCapturing Page 3: Forecasting...")
        await page.get_by_text("Forecasting", exact=False).first.click()
        await page.wait_for_timeout(3000)
        
        # Click Generate Forecast button
        gen_btn = page.get_by_role("button", name="GENERATE FORECAST").first
        if await gen_btn.count() > 0:
            await gen_btn.click()
            await page.wait_for_timeout(3500)

        await scroll_and_settle(page)
        await save_screenshot(page, "page_3_forecasting_full.png", full_page=True)
        await save_screenshot(page, "page_3_forecasting.png", full_page=True)

        # Focus: Forecast target metric & alert banner
        fc_target_card = page.locator(".gs-card").filter(has_text="NEXT-HOUR TARGET FORECAST").first
        if await fc_target_card.count() > 0:
            await save_screenshot(page, "feature_forecast_target_card.png", locator=fc_target_card)

        # -------------------------------------------------------------
        # PAGE 4: MODEL PERFORMANCE & BENCHMARK
        # -------------------------------------------------------------
        print("\nCapturing Page 4: Model Performance...")
        await page.get_by_text("Model Performance", exact=False).first.click()
        await page.wait_for_timeout(3000)
        await scroll_and_settle(page)
        await save_screenshot(page, "page_4_performance_full.png", full_page=True)
        await save_screenshot(page, "page_4_performance.png", full_page=True)

        # Tab: Actual vs. Predicted
        act_tab = page.get_by_role("tab", name="Actual vs. Predicted").first
        if await act_tab.count() > 0:
            await act_tab.click()
            await page.wait_for_timeout(2500)
            await scroll_and_settle(page)
            await save_screenshot(page, "feature_performance_actual_vs_predicted.png", full_page=False)

        # Tab: Feature Importance
        feat_tab = page.get_by_role("tab", name="Error Analysis & Feature Importance").first
        if await feat_tab.count() > 0:
            await feat_tab.click()
            await page.wait_for_timeout(2500)
            await scroll_and_settle(page)
            await save_screenshot(page, "feature_performance_feature_importance.png", full_page=False)

        # -------------------------------------------------------------
        # PAGE 5: DATA QUALITY & INTEGRITY
        # -------------------------------------------------------------
        print("\nCapturing Page 5: Data Quality...")
        await page.get_by_text("Data Quality", exact=False).first.click()
        await page.wait_for_timeout(3000)
        await scroll_and_settle(page)
        await save_screenshot(page, "page_5_quality_full.png", full_page=True)
        await save_screenshot(page, "page_5_quality.png", full_page=True)

        # Focus: Verification Checks Table
        table_section = page.locator("section.main").first
        if await table_section.count() > 0:
            await save_screenshot(page, "feature_quality_verification_table.png", full_page=False)

        # -------------------------------------------------------------
        # PAGE 6: ABOUT PROJECT & METHODOLOGY
        # -------------------------------------------------------------
        print("\nCapturing Page 6: About Project...")
        await page.get_by_text("About Project", exact=False).first.click()
        await page.wait_for_timeout(3000)
        await scroll_and_settle(page)
        await save_screenshot(page, "page_6_about_full.png", full_page=True)
        await save_screenshot(page, "page_6_about.png", full_page=True)

        await browser.close()
        print("\nAll comprehensive screenshots captured successfully!")

if __name__ == "__main__":
    asyncio.run(run_capture())
