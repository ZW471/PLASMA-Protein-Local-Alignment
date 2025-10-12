"""
Run utilities for directory and logging setup.
"""

import os
from datetime import datetime
from pathlib import Path
from loguru import logger
from typing import Optional


def create_run_directory(task: str, base_runs_dir: Optional[Path] = None) -> Path:
    """Create a timestamped run directory."""
    if base_runs_dir is None:
        # Default to plasma/runs
        base_runs_dir = Path(__file__).parent.parent / "runs"
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_runs_dir / f"{task}_{timestamp}"
    
    # Create directory
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def setup_logging(run_dir: Path, task: str) -> None:
    """Setup logging for the run."""
    log_file = run_dir / f"{task}.log"
    
    # Clear any existing handlers
    logger.remove()
    
    # Add console handler
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler
    logger.add(
        log_file,
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    logger.info(f"Starting run for task: {task}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")
    
    return log_file