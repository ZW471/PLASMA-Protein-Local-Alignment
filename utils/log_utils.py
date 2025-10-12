"""
Logging utilities for VenusX Plasma project.
Handles Hydra configuration and custom logging setup.
"""

import os
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from loguru import logger


def setup_hydra_logging(run_name: str, plasma_path: Path):
    """
    Get the Hydra output directory for logging setup.
    
    Args:
        run_name: Name for the run directory (e.g., 'motif_split0')
        plasma_path: Path to the plasma directory
    """
    # Get run directory from Hydra if available
    if HydraConfig.initialized():
        hydra_cfg = HydraConfig.get()
        run_dir = Path(hydra_cfg.runtime.output_dir)
    else:
        # Fallback to manual directory creation
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = plasma_path / "runs" / f"{run_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def get_hydra_output_dir():
    """Get the current Hydra output directory."""
    if HydraConfig.initialized():
        hydra_cfg = HydraConfig.get()
        return Path(hydra_cfg.runtime.output_dir)
    return None


def setup_custom_logging(output_dir: Path, log_name: str):
    """
    Setup custom logging configuration that works with Hydra.
    
    Args:
        output_dir: Directory where logs should be stored
        log_name: Name of the log file (without extension)
    """
    # Remove default logger
    logger.remove()
    
    # Add console logging with colors
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> <yellow>|</yellow> <level>{level: <8}</level> <yellow>|</yellow> <cyan>{name}</cyan>:<magenta>{function}</magenta>:<blue>{line}</blue> <yellow>-</yellow> <level>{message}</level>",
        level="INFO"
    )
    
    # Add file logging
    log_file = output_dir / f"{log_name}.log"
    logger.add(
        log_file,
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    logger.info(f"Logging setup complete. Log file: {log_file}")
    return log_file


def create_hydra_config_override(run_dir: Path):
    """
    Create Hydra configuration override to use custom output directory.
    
    Args:
        run_dir: Directory where Hydra should store its outputs
    """
    # Ensure .hydra directory exists in run directory
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable for Hydra output directory
    os.environ["HYDRA_RUNTIME_OUTPUT_DIR"] = str(run_dir)
    
    return hydra_dir


def log_hydra_config(cfg):
    """Log the Hydra configuration for debugging."""
    logger.info("=== Hydra Configuration ===")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    if HydraConfig.initialized():
        hydra_cfg = HydraConfig.get()
        logger.info(f"Hydra output directory: {hydra_cfg.runtime.output_dir}")
        logger.info(f"Hydra config name: {hydra_cfg.job.name}")


def configure_hydra_for_runs(run_name: str):
    """
    Configure Hydra to use runs directory structure.
    This should be called before the @hydra.main decorator is processed.
    
    Args:
        run_name: Base name for the run
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get plasma path from script location (runtime-inferable)
    plasma_path = Path(__file__).parent.parent
    run_dir = plasma_path / "runs" / f"{run_name}_{timestamp}"
    
    # Create the directory structure
    run_dir.mkdir(parents=True, exist_ok=True)
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables that Hydra will pick up
    os.environ["HYDRA_RUNTIME_OUTPUT_DIR"] = str(run_dir)
    
    return run_dir