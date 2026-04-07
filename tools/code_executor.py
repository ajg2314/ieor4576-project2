"""Tool: Sandboxed Python code execution for the EDA agent. (Grab Bag: Code Execution)

The EDA agent writes Python code (pandas, numpy, scipy, matplotlib) and submits
it here. We execute it in a subprocess with a timeout and capture stdout, stderr,
and any saved figure paths.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Hard timeout for agent-generated code
EXEC_TIMEOUT_SECONDS = 30


def execute_python(code: str) -> dict:
    """
    Execute agent-written Python code in a sandboxed subprocess.

    The agent should use matplotlib to save figures like:
        plt.savefig('artifacts/<name>.png', dpi=150, bbox_inches='tight')

    Args:
        code: Python source code string (agent-generated at runtime)

    Returns:
        dict with 'stdout', 'stderr', 'success', 'artifact_paths'
    """
    # Inject boilerplate so agent doesn't need to manage paths manually
    preamble = textwrap.dedent(f"""\
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
        import json, os, sys
        ARTIFACTS_DIR = '{ARTIFACTS_DIR}'
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    """)

    full_code = preamble + "\n" + code

    run_id = uuid.uuid4().hex[:8]
    script_path = ARTIFACTS_DIR / f"_script_{run_id}.py"
    script_path.write_text(full_code)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_SECONDS,
        )
        success = result.returncode == 0
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
    except subprocess.TimeoutExpired:
        success = False
        stdout = ""
        stderr = f"Code execution timed out after {EXEC_TIMEOUT_SECONDS}s"
    finally:
        script_path.unlink(missing_ok=True)

    # Discover any new artifacts written during this execution
    artifact_paths = [
        str(p.relative_to(Path(__file__).parent.parent))
        for p in ARTIFACTS_DIR.glob("*.png")
    ] + [
        str(p.relative_to(Path(__file__).parent.parent))
        for p in ARTIFACTS_DIR.glob("*.csv")
    ]

    return {
        "success": success,
        "stdout": stdout[:4000],  # truncate to avoid flooding context
        "stderr": stderr[:2000],
        "artifact_paths": artifact_paths,
    }
