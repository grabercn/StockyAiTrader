"""
Test runner for StockyAiTrader.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v           # Verbose output
    python run_tests.py -k "risk"    # Run only tests matching "risk"

Uses pytest under the hood. Install: pip install pytest
"""

import sys
import os

# Ensure StockyApps is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "StockyApps"))

if __name__ == "__main__":
    import pytest
    args = ["tests/", "-v", "--tb=short", "--no-header"] + sys.argv[1:]
    sys.exit(pytest.main(args))
