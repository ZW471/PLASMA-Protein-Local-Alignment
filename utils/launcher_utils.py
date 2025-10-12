"""
Launcher utilities for HPC job submission with Hydra integration.
Supports PBS/qsub job submission with configurable templates.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from string import Template
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class QsubLauncher:
    """PBS/Qsub job launcher with Hydra configuration support."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the qsub launcher with configuration.
        
        Args:
            config: Hydra configuration containing launcher settings
        """
        self.config = config
        self.launcher_config = config.get('launcher', {})
        
        # Set up directories
        self.script_dir = Path(self.launcher_config.get('script_dir', './qsub_scripts'))
        self.script_dir.mkdir(exist_ok=True)
        
        # Get current working directory
        self.work_dir = Path.cwd()
        
    def _format_job_name(self, base_config: Dict[str, Any]) -> str:
        """Generate job name from template and config."""
        template = self.launcher_config.get('job_name_template', '{task}_job')
        try:
            return template.format(**base_config)
        except Exception as e:
            logger.warning(f"Failed to format job name template: {e}")
            return f"hydra_job_{base_config.get('task', 'unknown')}"
    
    def _get_output_paths(self, job_name: str, base_config: Dict[str, Any]) -> tuple[str, str]:
        """Generate output and error file paths."""
        # Check if there's a Hydra output directory passed directly
        hydra_run_dir = base_config.get('hydra_output_dir')
        
        if not hydra_run_dir:
            # Check if there's a Hydra output directory in the nested config
            for key, value in base_config.items():
                if key == 'hydra' and isinstance(value, dict):
                    hydra_run_dir = value.get('runtime', {}).get('output_dir')
                    break
        
        if hydra_run_dir:
            # Use the Hydra output directory to maintain sweep structure
            output_dir = Path(hydra_run_dir)
            error_dir = Path(hydra_run_dir)
        else:
            # Fallback to configured directories
            output_dir = Path(self.launcher_config.get('output_dir', './qsub_logs'))
            error_dir = Path(self.launcher_config.get('error_dir', './qsub_logs'))
            
            # Include task subdirectory if specified
            task = base_config.get('task', '')
            if task:
                output_dir = output_dir / task
                error_dir = error_dir / task
        
        # Create directories if they don't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{job_name}.pbs.out"
        error_path = error_dir / f"{job_name}.pbs.err"
        
        return str(output_path), str(error_path)
    
    def _format_additional_options(self) -> str:
        """Format additional PBS options."""
        additional_options = self.launcher_config.get('additional_options', [])
        if not additional_options:
            return ""
        
        formatted_options = []
        for option in additional_options:
            if not option.startswith('#PBS'):
                option = f"#PBS {option}"
            formatted_options.append(option)
        
        return "\n".join(formatted_options)
    
    def _setup_conda_environment(self) -> str:
        """Generate conda environment setup commands."""
        # Check config for conda environment setting
        conda_env = self.launcher_config.get('conda_env', None)
        
        # If conda_env is explicitly set to null/None, don't use conda
        if conda_env is None:
            return "# Conda environment disabled"
        
        # If empty string, try to detect current environment
        if conda_env == "":
            conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'plasma')
            if conda_env == 'base':
                conda_env = 'plasma'
        
        # Check for custom conda init script
        custom_init = self.launcher_config.get('conda_init_script', None)
        
        if custom_init:
            return f"""
# Use custom conda initialization script
source {custom_init}
conda activate {conda_env}
"""
        
        return f"""
# Setup conda environment (try multiple common paths)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "/usr/local/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/usr/local/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    # conda is already in PATH
    :
else
    echo "Warning: conda not found in common locations"
    # Try to add common conda paths to PATH
    export PATH="$HOME/miniconda3/bin:$HOME/anaconda3/bin:/opt/conda/bin:$PATH"
fi

# Activate the specific environment
if command -v conda >/dev/null 2>&1; then
    conda activate {conda_env}
else
    echo "Error: conda command not available after setup"
    exit 1
fi
"""
    
    def _generate_qsub_script(self, command: str, job_name: str, 
                             base_config: Dict[str, Any]) -> str:
        """Generate PBS script from template."""
        
        output_path, error_path = self._get_output_paths(job_name, base_config)
        additional_pbs_options = self._format_additional_options()
        conda_setup = self._setup_conda_environment()
        
        # Get resources configuration
        resources = self.launcher_config.get('resources', {})
        
        # Template variables - flatten resources for easier access
        template_vars = {
            'queue': self.launcher_config.get('queue', 'default'),
            'walltime': self.launcher_config.get('walltime', '24:00:00'),
            'select': resources.get('select', 1),
            'ncpus': resources.get('ncpus', 1),
            'ngpus': resources.get('ngpus', 0),
            'mem': resources.get('mem', '8gb'),
            'job_name': job_name,
            'output_path': output_path,
            'error_path': error_path,
            'additional_pbs_options': additional_pbs_options,
            'work_dir': str(self.work_dir),
            'conda_setup': conda_setup,
            'command': command
        }
        
        # Use script template from config
        script_template = self.launcher_config.get('script_template', self._get_default_template())
        
        try:
            # Use string format instead of Template for more reliable substitution
            result = script_template.format(**template_vars)
            return result
        except Exception as e:
            logger.error(f"Failed to format script template: {e}")
            logger.error(f"Template variables: {template_vars}")
            logger.error(f"Template content: {script_template}")
            raise
    
    def _get_default_template(self) -> str:
        """Get default PBS script template."""
        return """#!/bin/bash
#PBS -q {queue}
#PBS -l walltime={walltime}
#PBS -l select={select}:ncpus={ncpus}:ngpus={ngpus}:mem={mem}
#PBS -N {job_name}
#PBS -o {output_path}
#PBS -e {error_path}
{additional_pbs_options}

# Change to working directory
cd {work_dir}

# Activate conda environment if specified
{conda_setup}

# Run the command
{command}
"""
    
    def submit_job(self, command: str, config: Dict[str, Any]) -> Optional[str]:
        """
        Submit a job to the qsub queue.
        
        Args:
            command: Command to execute
            config: Full configuration dictionary
            
        Returns:
            Job ID if successful, None otherwise
        """
        # Generate job name
        job_name = self._format_job_name(config)
        
        # Generate script content
        script_content = self._generate_qsub_script(command, job_name, config)
        
        # Check if debug mode is enabled
        if self.launcher_config.get('debug', False):
            print(f"\n{'='*60}")
            print(f"DEBUG MODE: Generated PBS script for job '{job_name}'")
            print(f"{'='*60}")
            print(script_content)
            print(f"{'='*60}")
            print(f"Command: {command}")
            print(f"{'='*60}\n")
            return None
        
        # Write script to file
        script_path = self.script_dir / f"{job_name}.pbs"
        script_path.write_text(script_content)
        logger.info(f"Generated PBS script: {script_path}")
        
        # Submit job if configured to do so
        if self.launcher_config.get('submit_immediately', True):
            try:
                result = subprocess.run(
                    ['qsub', str(script_path)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                job_id = result.stdout.strip()
                logger.info(f"Submitted job {job_id} for {job_name}")
                return job_id
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to submit job: {e}")
                logger.error(f"qsub stderr: {e.stderr}")
                return None
            except FileNotFoundError:
                logger.error("qsub command not found. Make sure PBS is installed and in PATH.")
                return None
        else:
            logger.info(f"Script generated but not submitted: {script_path}")
            return None


def create_launcher(config: Dict[str, Any]) -> QsubLauncher:
    """
    Factory function to create launcher based on configuration.
    
    Args:
        config: Hydra configuration
        
    Returns:
        Configured launcher instance
    """
    launcher_type = config.get('launcher', {}).get('type', 'qsub')
    
    if launcher_type == 'qsub':
        return QsubLauncher(config)
    else:
        raise ValueError(f"Unsupported launcher type: {launcher_type}")


# Hydra launcher plugin integration functions
def launch_job(job_cmd: List[str], job_config: Dict[str, Any]) -> Optional[str]:
    """
    Launch a single job using the configured launcher.
    
    Args:
        job_cmd: Command to execute as list of strings
        job_config: Job-specific configuration
        
    Returns:
        Job ID if successful, None otherwise
    """
    # Convert command list to string
    command = ' '.join(job_cmd)
    
    # Create launcher
    launcher = create_launcher(job_config)
    
    # Submit job
    return launcher.submit_job(command, job_config)


def launch_multirun_jobs(commands: List[List[str]], configs: List[Dict[str, Any]]) -> List[Optional[str]]:
    """
    Launch multiple jobs for multirun scenarios.
    
    Args:
        commands: List of commands to execute
        configs: List of configurations for each job
        
    Returns:
        List of job IDs (None for failed submissions)
    """
    job_ids = []
    
    for cmd, config in zip(commands, configs):
        job_id = launch_job(cmd, config)
        job_ids.append(job_id)
    
    return job_ids