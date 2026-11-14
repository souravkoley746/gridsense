"""UI smoke tests using Streamlit's AppTest across all 6 dashboard pages (Phase 7.6)."""
import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except Exception:  # pragma: no cover
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(AppTest is None, "Streamlit not installed")
@unittest.skipUnless((ROOT / "models" / "MT_001" / "metadata.json").exists(), "Requires trained MT_001 model")
class TestAppSmoke(unittest.TestCase):
    def test_all_six_pages_render_without_exception(self):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
        at.run()
        self.assertFalse(at.exception, f"Overview page raised exception: {at.exception}")

        six_pages = [
            "Historical Analysis",
            "Forecasting",
            "Model Performance",
            "Data Quality",
            "About Project",
            "Overview",
        ]
        for page in six_pages:
            at.radio(key="main_nav").set_value(page).run()
            self.assertFalse(at.exception, f"Page '{page}' raised exception: {at.exception}")


if __name__ == "__main__":
    unittest.main()
