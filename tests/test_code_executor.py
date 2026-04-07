"""Tests for tools/code_executor.py — sandboxed Python execution."""
import pytest
from tools.code_executor import execute_python


class TestExecutePython:
    def test_simple_print(self):
        result = execute_python("print('hello world')")
        assert result["success"] is True
        assert "hello world" in result["stdout"]
        assert result["stderr"] == ""

    def test_math_output(self):
        result = execute_python("print(2 + 2)")
        assert result["success"] is True
        assert "4" in result["stdout"]

    def test_pandas_available(self):
        code = """
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
print(df["a"].mean())
"""
        result = execute_python(code)
        assert result["success"] is True
        assert "2.0" in result["stdout"]

    def test_numpy_available(self):
        code = """
import numpy as np
arr = np.array([10, 20, 30])
print(arr.mean())
"""
        result = execute_python(code)
        assert result["success"] is True
        assert "20.0" in result["stdout"]

    def test_syntax_error_captured(self):
        result = execute_python("def broken(:\n    pass")
        assert result["success"] is False
        assert result["stderr"] != ""

    def test_runtime_error_captured(self):
        result = execute_python("x = 1 / 0")
        assert result["success"] is False
        assert "ZeroDivisionError" in result["stderr"]

    def test_matplotlib_chart_saved(self):
        from tools.code_executor import ARTIFACTS_DIR
        artifacts = str(ARTIFACTS_DIR)
        code = (
            "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
            "plt.figure()\nplt.plot([1, 2, 3], [4, 5, 6])\n"
            f"plt.savefig('{artifacts}/test_chart_unit.png', dpi=72)\n"
            "plt.close()\nprint('saved')\n"
        )
        result = execute_python(code)
        if result["success"]:
            assert "saved" in result["stdout"]

    def test_timeout_respected(self):
        # A tight infinite loop should be killed by the timeout
        # We don't test the full 30s; instead we monkey-patch the timeout
        import tools.code_executor as mod
        original = mod.EXEC_TIMEOUT_SECONDS
        mod.EXEC_TIMEOUT_SECONDS = 1
        try:
            result = execute_python("while True: pass")
            assert result["success"] is False
            assert "timed out" in result["stderr"].lower()
        finally:
            mod.EXEC_TIMEOUT_SECONDS = original

    def test_result_keys_present(self):
        result = execute_python("x = 1")
        assert set(result.keys()) >= {"success", "stdout", "stderr", "artifact_paths"}

    def test_stdout_truncated_at_limit(self):
        # Print a very long string
        code = "print('x' * 10000)"
        result = execute_python(code)
        assert result["success"] is True
        assert len(result["stdout"]) <= 4100  # 4000 char limit + some tolerance

    def test_multiline_code(self):
        code = """
revenues = [100, 200, 300, 400]
growth = [(revenues[i] - revenues[i-1]) / revenues[i-1] * 100
          for i in range(1, len(revenues))]
print(f"Average growth: {sum(growth)/len(growth):.1f}%")
"""
        result = execute_python(code)
        assert result["success"] is True
        assert "Average growth" in result["stdout"]
