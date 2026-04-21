"""Tests for upgraded LLM reasoner, deep analyze worker, window collapse."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "StockyApps"))


class TestLLMReasoner:
    def test_template_reasoning(self):
        from core.llm_reasoner import _template_reasoning
        result = _template_reasoning("AAPL", "BUY", 0.85, 150.0, 2.5,
                                      [0.05, 0.10, 0.85], "low", "bullish", "RSI_14, macd")
        assert "BUY" in result
        assert "AAPL" in result
        assert "150" in result

    # NOTE: test_generate_reasoning loads TinyLlama model (~30s on CPU).
    # Run manually with: pytest tests/test_llm_and_deep.py::TestLLMReasoner::test_generate_reasoning -v
    # def test_generate_reasoning_returns_string(self):
    #     from core.llm_reasoner import generate_reasoning
    #     result = generate_reasoning("TSLA", "SELL", 0.7, 200.0, 5.0, [0.70, 0.20, 0.10])
    #     assert "TSLA" in result

    def test_deep_template(self):
        from core.llm_reasoner import _deep_template
        result = _deep_template("NVDA", "BUY", 0.9, 300.0, 4.0,
                                 [0.05, 0.05, 0.90],
                                 {"RSI_14": 500, "macd": 300},
                                 50, 295.0, 310.0)
        assert "NVDA" in result
        assert "bullish" in result.lower()
        assert len(result) > 50

    # NOTE: generate_deep_analysis loads TinyLlama (~30s). Run manually.
    # def test_generate_deep_analysis(self): ...

    def test_model_id(self):
        from core.llm_reasoner import get_model_id
        mid = get_model_id()
        assert "TinyLlama" in mid

    def test_is_available_returns_bool(self):
        from core.llm_reasoner import is_available
        assert isinstance(is_available(), bool)

    # NOTE: cache test loads TinyLlama. Run manually.
    # def test_cache_works(self): ...

    def test_quality_gate_rejects_questions(self):
        """Template should never contain question marks."""
        from core.llm_reasoner import _template_reasoning
        result = _template_reasoning("X", "HOLD", 0.5, 50, 1, [0.3, 0.4, 0.3], "low", "neutral", "")
        assert "?" not in result


class TestModelManager:
    def test_managed_models_has_tinyllama(self):
        from core.model_manager import MANAGED_MODELS
        names = [m.name for m in MANAGED_MODELS]
        assert "TinyLlama-Chat" in names

    def test_managed_models_count(self):
        from core.model_manager import MANAGED_MODELS
        assert len(MANAGED_MODELS) >= 5

    def test_distilgpt2_not_required(self):
        from core.model_manager import MANAGED_MODELS
        for m in MANAGED_MODELS:
            if m.name == "DistilGPT2":
                assert not m.required

    def test_tinyllama_required(self):
        from core.model_manager import MANAGED_MODELS
        for m in MANAGED_MODELS:
            if m.name == "TinyLlama-Chat":
                assert m.required


class TestDeepAnalyzeWorker:
    def test_worker_creates(self):
        from panels.workers import _DeepAnalyzeWorker
        from core.scanner import ScanResult
        r = ScanResult("TEST", "BUY", 0.8, 100, 10, 95, 110, 2.0,
                        [0.1, 0.1, 0.8], {"RSI": 500}, "test", 0.8)
        w = _DeepAnalyzeWorker(r)
        assert w.r.ticker == "TEST"

    def test_worker_has_poll(self):
        from panels.workers import _DeepAnalyzeWorker
        from core.scanner import ScanResult
        r = ScanResult("TEST", "BUY", 0.8, 100, 10, 95, 110, 2.0,
                        [0.1, 0.1, 0.8], {}, "test", 0.8)
        w = _DeepAnalyzeWorker(r)
        assert hasattr(w, 'poll_progress')
        assert w.poll_progress() == []

    def test_emit_progress_queues(self):
        from panels.workers import _DeepAnalyzeWorker
        from core.scanner import ScanResult
        r = ScanResult("TEST", "BUY", 0.8, 100, 10, 95, 110, 2.0,
                        [0.1, 0.1, 0.8], {}, "test", 0.8)
        w = _DeepAnalyzeWorker(r)
        w._emit_progress(50, "Testing...", "detail")
        items = w.poll_progress()
        assert len(items) == 1
        assert items[0] == (50, "Testing...", "detail")
        # Queue should be empty after poll
        assert w.poll_progress() == []


class TestWindowCollapse:
    def test_import(self):
        from core.ui.window_collapse import WindowCollapse
        assert WindowCollapse is not None
