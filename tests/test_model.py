import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ROCCurveTool


class TestROCCurve:
    def test_compute_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
        tool = ROCCurveTool()
        results = tool.compute(y_true, y_score)
        assert results["auc"]["macro"] > 0.5
