"""Tests for core.tax_report — Form 8949 generation."""

import os
from core.tax_report import generate_form_8949, _format_text_report, _save_csv, REPORT_DIR
from core.broker import AlpacaBroker


class TestTaxReport:
    def test_no_trades_returns_empty(self):
        broker = AlpacaBroker("fake", "fake")
        result = generate_form_8949(broker, 2024)
        assert result["short_term"] == []
        assert result["long_term"] == []
        assert "no closed trades" in result["text"].lower() or result["summary"]["total_trades"] == 0

    def test_text_report_formatting(self):
        text = _format_text_report(
            short_term=[{
                "description": "10 shares AAPL",
                "date_acquired": "2024-06-01",
                "date_sold": "2024-06-15",
                "proceeds": 1500.00,
                "cost_basis": 1400.00,
                "gain_loss": 100.00,
                "holding_days": 14,
                "code": "B",
            }],
            long_term=[],
            summary={
                "tax_year": 2024,
                "total_trades": 1,
                "short_term_trades": 1,
                "long_term_trades": 0,
                "total_proceeds": 1500.00,
                "total_cost_basis": 1400.00,
                "total_gain_loss": 100.00,
                "short_term_gain_loss": 100.00,
                "long_term_gain_loss": 0.00,
            },
        )
        assert "FORM 8949" in text
        assert "SHORT-TERM" in text
        assert "AAPL" in text
        assert "1,500.00" in text

    def test_csv_creates_file(self, tmp_path, monkeypatch):
        import core.tax_report as tr
        monkeypatch.setattr(tr, "REPORT_DIR", str(tmp_path))

        path = _save_csv(
            short_term=[{
                "description": "5 shares TSLA",
                "date_acquired": "2024-03-01",
                "date_sold": "2024-03-10",
                "proceeds": 1000.00,
                "cost_basis": 900.00,
                "gain_loss": 100.00,
                "holding_days": 9,
                "wash_sale_adj": 0,
                "code": "B",
            }],
            long_term=[],
            summary={
                "total_proceeds": 1000.00,
                "total_cost_basis": 900.00,
                "total_gain_loss": 100.00,
            },
            tax_year=2024,
        )
        assert os.path.exists(path)
        content = open(path).read()
        assert "TSLA" in content
        assert "1000" in content
