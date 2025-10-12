"""
Time tracking utilities for evaluation and training scripts.

This module provides utilities for tracking execution time and calculating
timing statistics for training, evaluation, and baseline comparison scripts.
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class TimeTracker:
    """
    Class for tracking timing information during training/evaluation.
    
    Supports tracking total time, sample-level timing, and saving results.
    """
    
    def __init__(self, process_name: str = "process"):
        """
        Initialize time tracker.
        
        Args:
            process_name: Name of the process being tracked (e.g., "evaluation", "training")
        """
        self.process_name = process_name
        self.start_time = None
        self.end_time = None
        self.sample_times = []
        self.total_samples = 0
        
    def start(self):
        """Start timing the process."""
        self.start_time = time.time()
        logger.info(f"Started timing {self.process_name}")
        
    def end(self):
        """End timing the process."""
        self.end_time = time.time()
        logger.info(f"Finished timing {self.process_name}")
        
    def add_sample_time(self, sample_time: float, num_samples: int = 1):
        """
        Add timing for a batch of samples.
        
        Args:
            sample_time: Time taken to process the samples
            num_samples: Number of samples processed in this time
        """
        self.sample_times.append(sample_time)
        self.total_samples += num_samples
        
    def get_total_time(self) -> float:
        """Get total elapsed time in seconds (wall clock time)."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
        
    def get_actual_processing_time(self) -> float:
        """Get actual processing time by summing all sample times."""
        return sum(self.sample_times)
        
    def get_average_time_per_sample(self) -> float:
        """Get average time per sample in seconds based on actual processing time."""
        if self.total_samples == 0:
            return 0.0
        actual_processing_time = self.get_actual_processing_time()
        return actual_processing_time / self.total_samples
        
    def get_timing_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive timing summary with generic field names.
        
        Returns:
            Dictionary containing all timing statistics
        """
        wall_clock_time = self.get_total_time()
        actual_processing_time = self.get_actual_processing_time()
        avg_time_per_sample = self.get_average_time_per_sample()
        
        return {
            'total_time_seconds': actual_processing_time,  # Use actual processing time, not wall clock
            'total_time_minutes': actual_processing_time / 60,
            'total_time_hours': actual_processing_time / 3600,
            'wall_clock_time_seconds': wall_clock_time,  # Also include wall clock for reference
            'average_time_per_sample_seconds': avg_time_per_sample,
            'average_time_per_sample_milliseconds': avg_time_per_sample * 1000,
            'total_samples_processed': self.total_samples,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'created_at': datetime.now().isoformat()
        }
        
    def save_timing_data(self, output_path: Path):
        """
        Save timing data to JSON file.
        
        Args:
            output_path: Path where to save the timing data
        """
        timing_data = self.get_timing_summary()
        
        with open(output_path, 'w') as f:
            json.dump(timing_data, f, indent=2)
            
        logger.success(f"Timing data saved to {output_path}")
        logger.info(f"Total {self.process_name} processing time: {timing_data['total_time_seconds']:.2f} seconds")
        logger.info(f"Wall clock time: {timing_data['wall_clock_time_seconds']:.2f} seconds")
        logger.info(f"Average time per sample: {timing_data['average_time_per_sample_milliseconds']:.2f} ms")


def track_evaluation_time(func):
    """
    Decorator to automatically track evaluation time for functions.
    
    This decorator automatically creates a TimeTracker, starts timing before
    the function executes, ends timing after completion, and saves the results.
    
    Usage:
        @track_evaluation_time
        def evaluate_model(...):
            # evaluation code
            pass
    """
    def wrapper(*args, **kwargs):
        # Extract run_dir from args/kwargs if available
        run_dir = None
        if 'run_dir' in kwargs:
            run_dir = kwargs['run_dir']
        elif len(args) > 0 and hasattr(args[0], 'parent'):
            # Assume first arg might be a path
            run_dir = args[0] if isinstance(args[0], Path) else None
            
        timer = TimeTracker("evaluation")
        timer.start()
        
        try:
            result = func(*args, **kwargs)
        finally:
            timer.end()
            
            # Save timing data if run_dir is available
            if run_dir:
                timing_path = Path(run_dir) / "inference_time.json"
                timer.save_timing_data(timing_path)
            else:
                logger.warning("Could not determine output directory for timing data")
                
        return result
    return wrapper


def create_time_json(
    run_dir: Path,
    total_time: float,
    num_samples: int,
    filename: str = "inference_time.json",
    additional_data: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Create time JSON file with timing information using generic field names.
    
    Args:
        run_dir: Directory where to save the file
        total_time: Total time in seconds
        num_samples: Number of samples processed
        filename: Name of the JSON file to create (e.g., "training_time.json", "inference_time.json")
        additional_data: Additional data to include in the JSON
        
    Returns:
        Path to the created file
    """
    timing_data = {
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'total_time_hours': total_time / 3600,
        'average_time_per_sample_seconds': total_time / max(num_samples, 1),
        'average_time_per_sample_milliseconds': (total_time / max(num_samples, 1)) * 1000,
        'total_samples_processed': num_samples,
        'created_at': datetime.now().isoformat()
    }
    
    # Add any additional data
    if additional_data:
        timing_data.update(additional_data)
    
    output_path = run_dir / filename
    with open(output_path, 'w') as f:
        json.dump(timing_data, f, indent=2)
        
    logger.success(f"Time data saved to {output_path}")
    return output_path