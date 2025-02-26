import numpy as np
import random
from collections import deque

class IOAwareCPUAlgorithm:
    """
    Advanced power-aware scheduling algorithm that dynamically adjusts CPU frequency
    based on a multi-stage analysis of task properties, historical usage trends, 
    and predictive heuristics. Incorporates weighted history, decision entropy, and 
    a predictive stabilization mechanism to enhance adaptability.
    """

    def __init__(self, freqs, alpha=2.5, smoothing_factor=1.2, penalty_factor=0.2,
                 min_freq_weight=0.1, max_freq_weight=0.4, io_threshold=10, history_size=100,
                 entropy_weight=0.3, prediction_factor=0.4, outlier_rejection=True):
        """
        Parameters:
        - freqs: List of available CPU frequencies
        - alpha: Power scaling factor
        - smoothing_factor: Exponential smoothing weight for historical I/O trends
        - penalty_factor: Additional penalty for high I/O tasks
        - min_freq_weight: Multiplier for minimum frequency scaling
        - max_freq_weight: Multiplier for maximum frequency scaling
        - io_threshold: Dynamic threshold for classifying I/O-intensive tasks
        - history_size: Number of past tasks to consider for trend analysis
        - entropy_weight: Weighting factor for decision entropy consideration
        - prediction_factor: Weighting factor for predictive adjustments
        - outlier_rejection: If True, removes extreme I/O readings before processing
        """
        self.freqs = sorted(freqs)
        self.alpha = alpha
        self.smoothing_factor = smoothing_factor
        self.penalty_factor = penalty_factor
        self.min_freq_weight = min_freq_weight
        self.max_freq_weight = max_freq_weight
        self.io_threshold = io_threshold
        self.history_size = history_size
        self.entropy_weight = entropy_weight
        self.prediction_factor = prediction_factor
        self.outlier_rejection = outlier_rejection
        self.task_history = deque(maxlen=history_size)
        self.smoothed_io_ratio = 0.5  # Start with a neutral baseline

    def _moving_average_io(self):
        """Compute a simple moving average of past task I/O intensities."""
        if not self.task_history:
            return 0.5
        return sum(self.task_history) / len(self.task_history)

    def _quadratic_mean_io(self):
        """Compute quadratic mean to bias high I/O task's influence."""
        if not self.task_history:
            return 0.5
        return (sum(x**2 for x in self.task_history) / len(self.task_history)) ** 0.5

    def _exponential_smoothing(self, new_io_ratio):
        """Apply exponential smoothing to track trends in I/O intensity."""
        self.smoothed_io_ratio = (
            self.smoothing_factor * new_io_ratio +
            (1 - self.smoothing_factor) * self.smoothed_io_ratio
        )
        return self.smoothed_io_ratio

    def _adaptive_threshold(self):
        """Dynamically adjust the I/O threshold based on past trends."""
        trend = self._quadratic_mean_io()
        return self.io_threshold * (1 + (trend - 0.5))

    def _compute_entropy(self):
        if not self.task_history:
            return 0.1  # Low entropy when there's no history
        probabilities = np.array(self.task_history) / sum(self.task_history)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        return self.entropy_weight * entropy

    def _predictive_adjustment(self, last_io, current_io):
        """Prediction step that adjusts based on a trend model."""
        trend_factor = (current_io - last_io) * self.prediction_factor
        return max(0, min(1, 0.5 + trend_factor))

    def adjust_frequency(self, task, freqs):
        """
        Dynamically adjust CPU frequency:
        - Uses I/O intensity, entropy, historical smoothing, and predictive adjustments.
        """
        io_ratio = task.io_interrupts / max(1, task.remaining_time + task.io_interrupts)

        # Outlier rejection: Remove extreme I/O values if enabled
        if self.outlier_rejection:
            self.task_history = deque(sorted(self.task_history)[1:-1], maxlen=self.history_size)

        # Update history
        self.task_history.append(io_ratio)

        # Smoothed I/O ratio and entropy calculations
        smoothed_io = self._exponential_smoothing(io_ratio)
        entropy_correction = self._compute_entropy()
        adaptive_threshold = self._adaptive_threshold()

        # Predictive adjustment
        predicted_adjustment = self._predictive_adjustment(
            self.task_history[-2] if len(self.task_history) > 1 else 0.5, 
            smoothed_io
        )

        # Frequency scaling factor computation
        if io_ratio > adaptive_threshold:
            scaling_factor = (self.min_freq_weight + (1 - smoothed_io) * self.penalty_factor) * predicted_adjustment
        else:
            scaling_factor = (self.max_freq_weight - smoothed_io * self.penalty_factor) * (1 - predicted_adjustment)

        # Apply entropy correction
        scaling_factor *= (1 - entropy_correction)

        # Normalize and compute target frequency
        scaling_factor = max(0.5, min(2.0, scaling_factor))
        target_freq = int(np.interp(scaling_factor, [0.5, 2.0], [min(self.freqs), max(self.freqs)]))

        return target_freq

