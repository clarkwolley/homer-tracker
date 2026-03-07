"""Tests for model evaluation utilities."""

import numpy as np
import pytest

from src.models.evaluate import evaluate_model


class TestEvaluateModel:
    def test_perfect_predictions(self):
        y_true = [0, 0, 0, 1, 1]
        y_prob = [0.1, 0.1, 0.2, 0.9, 0.8]
        metrics = evaluate_model(y_true, y_prob, threshold=0.5)
        assert metrics["roc_auc"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_coin_flip_predictions(self):
        np.random.seed(42)
        y_true = [0] * 50 + [1] * 50
        y_prob = np.random.uniform(0, 1, 100)
        metrics = evaluate_model(y_true, y_prob, threshold=0.5)
        # AUC for random should be around 0.5 (± noise)
        assert 0.3 <= metrics["roc_auc"] <= 0.7

    def test_output_keys(self):
        metrics = evaluate_model([0, 1], [0.3, 0.8])
        expected_keys = [
            "roc_auc", "brier_score", "precision", "recall",
            "f1", "threshold", "total_samples", "actual_positives",
            "predicted_positives",
        ]
        for key in expected_keys:
            assert key in metrics

    def test_threshold_affects_predictions(self):
        y_true = [0, 0, 1, 1]
        y_prob = [0.05, 0.08, 0.12, 0.15]

        high_thresh = evaluate_model(y_true, y_prob, threshold=0.5)
        low_thresh = evaluate_model(y_true, y_prob, threshold=0.10)

        # Low threshold predicts more positives
        assert low_thresh["predicted_positives"] >= high_thresh["predicted_positives"]

    def test_brier_score_perfect(self):
        metrics = evaluate_model([1, 0], [1.0, 0.0])
        assert metrics["brier_score"] == 0.0

    def test_sample_counts(self):
        y_true = [0, 0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.3, 0.8, 0.9]
        metrics = evaluate_model(y_true, y_prob)
        assert metrics["total_samples"] == 5
        assert metrics["actual_positives"] == 2
