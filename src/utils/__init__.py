"""
Utility modules for xLAM qLoRA training.
"""

from .data_quality import validate_example, compute_statistics, filter_by_length
from .training_utils import (
    LossLoggingCallback,
    GradientMonitorCallback,
    EarlyStoppingCallback,
    find_optimal_batch_size,
    print_training_summary,
    setup_dataloader_optimizations,
)
from .batched_inference import (
    generate_batch,
    generate_batch_with_cache,
    estimate_batch_size_for_inference,
)

__all__ = [
    # Data quality
    "validate_example",
    "compute_statistics",
    "filter_by_length",
    # Training utils
    "LossLoggingCallback",
    "GradientMonitorCallback",
    "EarlyStoppingCallback",
    "find_optimal_batch_size",
    "print_training_summary",
    "setup_dataloader_optimizations",
    # Batched inference
    "generate_batch",
    "generate_batch_with_cache",
    "estimate_batch_size_for_inference",
]
