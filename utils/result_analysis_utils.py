#!/usr/bin/env python3
"""
Utility functions for loading and analyzing evaluation results from different methods.

This module provides functions to load evaluation results from various methods
(PLASMA, PLASMA-PF, EBA, backbone baselines, alignment baselines) and extract
standardized metrics for table generation.
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict


def find_evaluation_files(directory: str, pattern: str = "*evaluation_results.json") -> List[Path]:
    """Find all evaluation result files in a directory recursively.
    
    Args:
        directory: Root directory to search
        pattern: File pattern to match
        
    Returns:
        List of paths to evaluation result files
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Warning: Directory does not exist: {directory}")
        return []
    
    return list(directory.rglob(pattern))


def find_config_files(directory: str, pattern: str = "config.json") -> List[Path]:
    """Find all config files in a directory recursively.
    
    Args:
        directory: Root directory to search
        pattern: File pattern to match
        
    Returns:
        List of paths to config files
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Warning: Directory does not exist: {directory}")
        return []
    
    return list(directory.rglob(pattern))


def load_plasma_results(directory: str) -> Dict[str, Any]:
    """Load results from PLASMA training runs.
    
    Expected structure: {task}/{backbone}/{split}/evaluation_results.json
    
    Args:
        directory: Path to PLASMA training results
        
    Returns:
        Dictionary organized by task -> backbone -> split -> test_set -> metrics
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    eval_files = find_evaluation_files(directory)
    
    for eval_file in eval_files:
        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)
            
            # Extract task, backbone, and split from file path or data
            model_info = data.get('model_info', {})
            task = model_info.get('task', '')
            backbone = model_info.get('backbone_model', '')
            split = model_info.get('split', 0)
            
            # If not in model_info, try to parse from path
            if not task or not backbone:
                parts = eval_file.parts
                for i, part in enumerate(parts):
                    if part in ['active_site', 'binding_site', 'motif']:
                        task = part
                        if i + 1 < len(parts):
                            backbone = parts[i + 1]
                        break
            
            # Extract split from filename if not in model_info
            if split == 0 and 'split' in eval_file.stem:
                try:
                    split_part = [p for p in eval_file.stem.split('_') if 'split' in p][0]
                    split = int(split_part.replace('split', ''))
                except (IndexError, ValueError):
                    split = 0
            
            # Process both test sets
            for test_set in ['test_frequent', 'test_hard']:
                if test_set in data:
                    test_data = data[test_set].get('metrics', {})
                    
                    # Extract metrics with alignment_trained key
                    metrics = {}
                    for metric_name, metric_data in test_data.items():
                        if isinstance(metric_data, dict) and 'alignment_trained' in metric_data:
                            metrics[metric_name] = metric_data['alignment_trained']
                    
                    results[task][backbone][split][test_set] = metrics
                    
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not process PLASMA file {eval_file}: {e}")
            continue
    
    return dict(results)


def load_plasma_pf_results(directory: str) -> Dict[str, Any]:
    """Load results from PLASMA-PF evaluation runs.
    
    Expected structure: {task}/{backbone}/{split}/*_pf_evaluation_results.json
    
    Args:
        directory: Path to PLASMA-PF evaluation results
        
    Returns:
        Dictionary organized by task -> backbone -> split -> test_set -> metrics
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    eval_files = find_evaluation_files(directory, "*_pf_evaluation_results.json")
    
    for eval_file in eval_files:
        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)
            
            # Parse task, backbone, and split from filename
            filename = eval_file.stem
            parts = filename.split('_')
            
            # Extract task and split from filename
            # Handle compound task names like "active_site" -> should be "active_site", not "active"
            task = None
            split = None
            
            # Look for complete task names in filename
            full_filename = eval_file.stem
            for task_name in ['active_site', 'binding_site', 'motif']:
                if task_name in full_filename:
                    task = task_name
                    break
            
            # If no complete task name found, try first part
            if not task and parts:
                task = parts[0]
            
            # Extract split
            for part in parts:
                if 'split' in part:
                    try:
                        split = int(part.replace('split', ''))
                    except ValueError:
                        pass
                    break
            
            if split is None:
                print(f"Warning: Could not extract split from filename {filename}")
                continue
            
            # Extract backbone from parent directory (skip the numbered directory)
            # Structure is: .../task/backbone/split_number/file.json
            backbone = eval_file.parent.parent.name
            
            # Process both test sets
            for test_set in ['test_frequent', 'test_hard']:
                if test_set in data:
                    test_data = data[test_set].get('metrics', {})
                    
                    # Extract metrics with no_learning key (PLASMA-PF uses pre-trained features)
                    metrics = {}
                    for metric_name, metric_data in test_data.items():
                        if isinstance(metric_data, dict) and 'no_learning' in metric_data:
                            metrics[metric_name] = metric_data['no_learning']
                    
                    results[task][backbone][split][test_set] = metrics
                    
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not process PLASMA-PF file {eval_file}: {e}")
            continue
    
    return dict(results)


def load_eba_results(directory: str) -> Dict[str, Any]:
    """Load results from EBA evaluation runs.
    
    EBA results are stored like PLASMA-PF but can be identified by alignment_module=eba in config.
    
    Args:
        directory: Path to EBA evaluation results
        
    Returns:
        Dictionary organized by task -> backbone -> split -> test_set -> metrics
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    eval_files = find_evaluation_files(directory, "*_pf_evaluation_results.json")
    
    for eval_file in eval_files:
        try:
            # Assume all files in EBA directory are EBA runs
            # Load evaluation results
            with open(eval_file, 'r') as f:
                data = json.load(f)
            
            # Parse task, backbone, and split from filename (same as PLASMA-PF)
            filename = eval_file.stem
            parts = filename.split('_')
            
            # Extract task and split from filename
            # Handle compound task names like "active_site" -> should be "active_site", not "active"
            task = None
            split = None
            
            # Look for complete task names in filename
            full_filename = eval_file.stem
            for task_name in ['active_site', 'binding_site', 'motif']:
                if task_name in full_filename:
                    task = task_name
                    break
            
            # If no complete task name found, try first part
            if not task and parts:
                task = parts[0]
            
            # Extract split
            for part in parts:
                if 'split' in part:
                    try:
                        split = int(part.replace('split', ''))
                    except ValueError:
                        pass
                    break
            
            if split is None:
                print(f"Warning: Could not extract split from filename {filename}")
                continue
            
            # Extract backbone from parent directory (skip the numbered directory)
            # Structure is: .../task/backbone/split_number/file.json
            backbone = eval_file.parent.parent.name
            
            # Process both test sets
            for test_set in ['test_frequent', 'test_hard']:
                if test_set in data:
                    test_data = data[test_set].get('metrics', {})
                    
                    # Extract metrics with no_learning key (EBA uses pre-trained features like PLASMA-PF)
                    metrics = {}
                    for metric_name, metric_data in test_data.items():
                        if isinstance(metric_data, dict) and 'no_learning' in metric_data:
                            metrics[metric_name] = metric_data['no_learning']
                    
                    results[task][backbone][split][test_set] = metrics
                    
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not process EBA file {eval_file}: {e}")
            continue
    
    return dict(results)


def load_baseline_results(directory: str) -> Dict[str, Any]:
    """Load results from baseline evaluations.
    
    Expected structure: all/any/0/all_evaluation_results.json containing all splits
    
    Args:
        directory: Path to baseline evaluation results
        
    Returns:
        Dictionary organized by task -> backbone -> split -> test_set -> metrics
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    eval_files = find_evaluation_files(directory, "all_evaluation_results.json")
    
    for eval_file in eval_files:
        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)
            
            # Process each task in the data
            for task, task_data in data.items():
                if isinstance(task_data, dict):
                    # Process each split within the task data (split_0, split_1, split_2)
                    for split_key, split_data in task_data.items():
                        if split_key.startswith('split_') and isinstance(split_data, dict):
                            try:
                                split = int(split_key.split('_')[1])
                            except (IndexError, ValueError):
                                continue
                            
                            # Process both test sets
                            for test_set in ['test_frequent', 'test_hard']:
                                if test_set in split_data:
                                    test_data = split_data[test_set].get('metrics', {})
                                    
                                    # Process each metric
                                    for metric_name, metric_data in test_data.items():
                                        if isinstance(metric_data, dict):
                                            # Process each backbone model
                                            for backbone, value in metric_data.items():
                                                if backbone not in results[task]:
                                                    results[task][backbone] = defaultdict(dict)
                                                if split not in results[task][backbone]:
                                                    results[task][backbone][split] = defaultdict(dict)
                                                if test_set not in results[task][backbone][split]:
                                                    results[task][backbone][split][test_set] = {}
                                                
                                                results[task][backbone][split][test_set][metric_name] = value
                    
        except (json.JSONDecodeError, KeyError, FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not process baseline file {eval_file}: {e}")
            continue
    
    return dict(results)


def load_alignment_baseline_results(directory: str) -> Dict[str, Any]:
    """Load results from alignment baseline evaluations (Foldseek, TM-Align).
    
    Expected structure: {task}/{method}/{split}/*_evaluation.json
    
    Args:
        directory: Path to alignment baseline results
        
    Returns:
        Dictionary organized by task -> method -> split -> test_set -> metrics
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    eval_files = find_evaluation_files(directory, "*_evaluation.json")
    
    for eval_file in eval_files:
        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)
            
            # Parse task, method, and split from file path
            parts = eval_file.parts
            task = None
            method = None
            split = None
            
            # Find task in path
            for part in parts:
                if part in ['active_site', 'binding_site', 'motif']:
                    task = part
                    break
            
            # Method is the grandparent directory of the split directory
            # Structure is: .../task/method/split_number/file.json
            if len(parts) >= 3:
                method = parts[-3]  # Grandparent of the numbered directory
            
            # Split is from the filename
            filename = eval_file.stem
            filename_parts = filename.split('_')
            for part in filename_parts:
                if 'split' in part:
                    try:
                        split = int(part.replace('split', ''))
                    except ValueError:
                        pass
                    break
            
            if not all([task, method, split is not None]):
                print(f"Warning: Could not parse task/method/split from {eval_file}")
                continue
            
            # Process both test sets
            for test_set in ['test_frequent', 'test_hard']:
                if test_set in data:
                    test_data = data[test_set].get('metrics', {})
                    
                    # For alignment baselines, metrics are directly in test_data
                    metrics = {}
                    for metric_name, value in test_data.items():
                        if isinstance(value, (int, float)):
                            metrics[metric_name] = value
                    
                    results[task][method][split][test_set] = metrics
                    
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not process alignment baseline file {eval_file}: {e}")
            continue
    
    return dict(results)


def aggregate_splits(results: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate results across splits to compute mean and standard error.
    
    Args:
        results: Dictionary with split-level results
        
    Returns:
        Dictionary with aggregated statistics (mean, std, se)
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for task, task_data in results.items():
        for method, method_data in task_data.items():
            for test_set in ['test_frequent', 'test_hard']:
                # Collect all split values for each metric
                metric_values = defaultdict(list)
                
                for split, split_data in method_data.items():
                    if test_set in split_data:
                        for metric, value in split_data[test_set].items():
                            if value is not None:
                                metric_values[metric].append(value)
                
                # Calculate statistics for each metric
                for metric, values in metric_values.items():
                    if values:
                        mean = np.mean(values)
                        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
                        se = std / np.sqrt(len(values)) if len(values) > 0 else 0.0
                        
                        aggregated[task][method][test_set][metric] = {
                            'mean': mean,
                            'std': std,
                            'se': se,
                            'n': len(values),
                            'values': values
                        }
    
    return dict(aggregated)


def format_metric_value(mean: float, se: float, decimals: int = 2, se_decimals: int = None) -> str:
    """Format metric value as mean with subscript standard error using LaTeX pm.
    
    Args:
        mean: Mean value
        se: Standard error
        decimals: Number of decimal places for mean
        se_decimals: Number of decimal places for standard error (defaults to decimals)
        
    Returns:
        Formatted string like ".123_{\\pm.045}" (subscript format with LaTeX pm, without leading zero)
    """
    if mean is None:
        return ""
    
    if se_decimals is None:
        se_decimals = decimals
    
    # Format with specified decimal places and remove leading zero
    mean_str = f"{mean:.{decimals}f}"
    if mean_str.startswith("0."):
        mean_str = mean_str[1:]  # Remove leading zero
    
    # Always show standard error, must not be None
    if se is not None:
        se_str = f"{se:.{se_decimals}f}"
        if se_str.startswith("0."):
            se_str = se_str[1:]  # Remove leading zero
    else:
        raise ValueError(f"Standard error cannot be None for mean {mean}. Must have at least 2 splits for proper statistics.")
    
    return f"{mean_str}_{{\\pm{se_str}}}"


def get_cell_color(value: float, all_values: Dict[str, float], method: str, metric: str, color_scaling: Optional[Dict[str, Any]] = None, global_min_max: Optional[Dict[str, float]] = None) -> str:
    """Get LaTeX cell color command based on relative position between best and worst performers.
    
    Args:
        value: Current method's value
        all_values: Dictionary of {method: value} for all methods at this position
        method: Current method name
        metric: Metric name for special handling
        color_scaling: Optional color scaling configuration (method, exponential_base)
        global_min_max: Optional dict with 'min' and 'max' values across all backbone models
        
    Returns:
        LaTeX color command with intensity based on relative position (best=green!50, worst=red!50)
    """
    if value is None:
        return ""
    
    # Get color configuration
    if color_scaling and 'colors' in color_scaling:
        better_color = color_scaling['colors'].get('better_color', 'green')
        worse_color = color_scaling['colors'].get('worse_color', 'red') 
        neutral_color = color_scaling['colors'].get('neutral_color', 'gray')
        min_intensity = color_scaling['colors'].get('min_intensity', 10)
        max_intensity = color_scaling['colors'].get('max_intensity', 50)
        color_format = color_scaling['colors'].get('color_format', 'named')
    else:
        better_color, worse_color, neutral_color = 'green', 'red', 'gray'
        min_intensity, max_intensity = 10, 50
        color_format = 'named'
    
    def format_color(color, intensity):
        """Format color based on the color format setting."""
        if color_format == 'rgb':
            # For RGB colors, we need to handle intensity by mixing with white
            # Parse RGB values from string like "{255,153,0}"
            rgb_values = color.strip('{}').split(',')
            if len(rgb_values) == 3:
                try:
                    r, g, b = [int(x.strip()) for x in rgb_values]
                    # Mix with white based on intensity (lower intensity = more white mixing)
                    # intensity ranges from min_intensity to max_intensity
                    mix_factor = (intensity - min_intensity) / (max_intensity - min_intensity)
                    mixed_r = int(r * mix_factor + 255 * (1 - mix_factor))
                    mixed_g = int(g * mix_factor + 255 * (1 - mix_factor))
                    mixed_b = int(b * mix_factor + 255 * (1 - mix_factor))
                    return f"\\cellcolor[RGB]{{{mixed_r},{mixed_g},{mixed_b}}}"
                except ValueError:
                    # Fallback to neutral if parsing fails
                    return f"\\cellcolor[RGB]{{200,200,200}}"
            else:
                # Fallback to neutral if format is wrong
                return f"\\cellcolor[RGB]{{200,200,200}}"
        else:
            # For named colors, use the standard approach
            return f"\\cellcolor{{{color}!{intensity:.0f}}}"
    
    # For LMS metric, only compare PLASMA and PLASMA-PF (no TM-Align baseline)
    if metric == 'label_match_score':
        valid_values = {m: v for m, v in all_values.items() if v is not None and m in ['plasma', 'plasma_pf']}
        if len(valid_values) == 2:
            plasma_val = valid_values.get('plasma')
            plasma_pf_val = valid_values.get('plasma_pf')
            
            if method == 'plasma' and plasma_val is not None and plasma_pf_val is not None:
                if plasma_val > plasma_pf_val:
                    return format_color(better_color, max_intensity)  # Best performer
                else:
                    return format_color(neutral_color, min_intensity)
            elif method == 'plasma_pf' and plasma_val is not None and plasma_pf_val is not None:
                if plasma_pf_val > plasma_val:
                    return format_color(better_color, max_intensity)  # Best performer
                else:
                    return format_color(neutral_color, min_intensity)
        return ""
    
    # For all other metrics, use relative scaling based on range from best to worst
    tm_align_value = all_values.get('tm_align')
    if tm_align_value is None:
        return ""
    
    # Use global min/max if provided, otherwise use local values
    if global_min_max is not None:
        best_value = global_min_max['max']
        worst_value = global_min_max['min']
    else:
        # Get all valid values for scaling
        valid_values = [v for v in all_values.values() if v is not None]
        if len(valid_values) < 2:
            return format_color(neutral_color, min_intensity)
        
        # Find best and worst performers
        best_value = max(valid_values)
        worst_value = min(valid_values)
    
    # Handle edge case where all values are the same
    if best_value == worst_value:
        return format_color(neutral_color, min_intensity)
    
    # TM-Align baseline gets neutral gray
    if method == 'tm_align':
        return format_color(neutral_color, min_intensity)
    
    # Calculate relative position between worst (0) and best (1)
    relative_pos = (value - worst_value) / (best_value - worst_value)
    
    # Calculate position relative to TM-Align
    tm_align_pos = (tm_align_value - worst_value) / (best_value - worst_value)
    
    if relative_pos > tm_align_pos:
        # Better than TM-Align: green intensity based on how close to best
        diff_from_baseline = (relative_pos - tm_align_pos) / (1 - tm_align_pos)
        
        # Apply scaling method
        if color_scaling and color_scaling.get('method') == 'exponential':
            base = color_scaling.get('exponential_base', 2.0)
            # Use exponential scaling: e^(base * difference) - 1
            # Normalize by e^base - 1 to keep in [0,1] range
            scaled_diff = (np.exp(base * diff_from_baseline) - 1) / ((np.exp(base) - 1) + 1e-8) 
        else:
            scaled_diff = diff_from_baseline
            
        # Scale from min to max intensity
        intensity_range = max_intensity - min_intensity
        intensity = min_intensity + scaled_diff * intensity_range
        intensity = min(max(intensity, min_intensity), max_intensity)
        return format_color(better_color, intensity)
    elif relative_pos < tm_align_pos:
        # Worse than TM-Align: red intensity based on how close to worst  
        diff_from_baseline = (tm_align_pos - relative_pos) / tm_align_pos
        
        scaled_diff = diff_from_baseline
            
        # Scale from min to max intensity
        intensity_range = max_intensity - min_intensity
        intensity = min_intensity + scaled_diff * intensity_range
        intensity = min(max(intensity, min_intensity), max_intensity)
        return format_color(worse_color, intensity)
    else:
        # Equal to TM-Align: neutral gray
        return format_color(neutral_color, min_intensity)


def format_colored_cell(mean: float, se: float, all_values: Dict[str, float], method: str, metric: str, decimals: int = 2, se_decimals: int = None, color_scaling: Optional[Dict[str, Any]] = None, global_min_max: Optional[Dict[str, float]] = None) -> str:
    """Format metric value with color coding based on relative ranking.
    
    Args:
        mean: Mean value
        se: Standard error
        all_values: Dictionary of {method: value} for all methods at this position
        method: Current method name
        metric: Metric name for special handling
        decimals: Number of decimal places for mean
        se_decimals: Number of decimal places for standard error (defaults to decimals)
        color_scaling: Optional color scaling configuration (method, exponential_base)
        global_min_max: Optional dict with global min/max values
        
    Returns:
        Formatted LaTeX string with color coding
    """
    if mean is None:
        return ""
    
    if se_decimals is None:
        se_decimals = decimals
    
    # Get the basic formatted value
    formatted_value = format_metric_value(mean, se, decimals, se_decimals)
    
    # Check if this should be bolded (best value)
    should_bold = False
    if color_scaling and color_scaling.get('text_formatting', {}).get('bold_best', False):
        # Get comparison decimals (default: 2)
        bold_comparison_decimals = color_scaling.get('text_formatting', {}).get('bold_comparison_decimals', 2)
        
        # Define method priority for tie-breaking: PLASMA > PLASMA-PF > EBA > Backbone > Foldseek > TM-Align
        method_priority = {
            'plasma': 1,
            'plasma_pf': 2, 
            'eba': 3,
            'backbone': 4,
            'foldseek': 5,
            'tm_align': 6
        }
        
        # Don't bold Foldseek and TM-Align baseline methods
        if method not in ['foldseek', 'tm_align']:
            # For LMS metric, only compare PLASMA and PLASMA-PF
            if metric == 'label_match_score':
                lms_methods = ['plasma', 'plasma_pf']
                valid_methods = {k: v for k, v in all_values.items() if v is not None and k in lms_methods}
            else:
                # For other metrics, compare all non-baseline methods
                valid_methods = {k: v for k, v in all_values.items() if v is not None}
            
            if valid_methods:
                # Round values to specified decimal places for comparison
                rounded_values = {k: round(v, bold_comparison_decimals) for k, v in valid_methods.items()}
                max_rounded_value = max(rounded_values.values())
                
                # Find all methods with the maximum rounded value
                best_methods = [k for k, v in rounded_values.items() if v == max_rounded_value]
                
                # If current method's rounded value equals the max, check priority
                if round(mean, bold_comparison_decimals) == max_rounded_value:
                    # Among tied methods, choose the one with highest priority (lowest number)
                    best_method = min(best_methods, key=lambda x: method_priority.get(x, 999))
                    if method == best_method:
                        should_bold = True
    
    # Apply bolding if needed
    if should_bold:
        formatted_value = f"\\mathbf{{{formatted_value}}}"
    
    # Get color command
    color_cmd = get_cell_color(mean, all_values, method, metric, color_scaling, global_min_max)
    
    # Combine color and value
    if color_cmd:
        return f"{color_cmd}${formatted_value}$"
    else:
        return f"${formatted_value}$"


def calculate_task_min_max(all_results: Dict[str, Dict[str, Any]], task: str, backbones: List[str], test_set: str, metric: str, methods: List[str]) -> Dict[str, float]:
    """Calculate minimum and maximum values for a specific task across all backbone combinations.
    
    Args:
        all_results: All loaded results
        task: Task name
        backbones: List of backbone model names
        test_set: Test set name
        metric: Metric name
        methods: List of method names to consider
        
    Returns:
        Dictionary with 'min' and 'max' values for this task
    """
    all_values = []
    
    for backbone in backbones:
        method_values = get_all_method_values(all_results, task, backbone, test_set, metric)
        # Include values from all specified methods (including foldseek, tm_align)
        for method in methods:
            if method in method_values and method_values[method] is not None:
                all_values.append(method_values[method])
    
    # Also include alignment baseline values for this task (they don't have backbone variants)
    if 'foldseek' in methods:
        foldseek_values = get_all_method_values(all_results, task, 'Foldseek', test_set, metric)
        if 'foldseek' in foldseek_values and foldseek_values['foldseek'] is not None:
            all_values.append(foldseek_values['foldseek'])
    
    if 'tm_align' in methods:
        tm_align_values = get_all_method_values(all_results, task, 'TM-Align', test_set, metric)
        if 'tm_align' in tm_align_values and tm_align_values['tm_align'] is not None:
            all_values.append(tm_align_values['tm_align'])
    
    if not all_values:
        return {'min': 0.0, 'max': 1.0}
    
    return {'min': min(all_values), 'max': max(all_values)}


def get_all_method_values(all_results: Dict[str, Dict[str, Any]], task: str, backbone: str, test_set: str, metric: str, method: str = None) -> Dict[str, float]:
    """Get all method values for a specific position to enable relative ranking.
    
    Args:
        all_results: All loaded results
        task: Task name
        backbone: Backbone model name
        test_set: Test set name
        metric: Metric name
        method: Current method name (for alignment baseline detection)
        
    Returns:
        Dictionary mapping method names to their mean values
    """
    method_values = {}
    
    # For alignment baselines, they don't have backbone-specific values
    # Instead, they span all columns, so we only compare them with each other
    if method in ['foldseek', 'tm_align']:
        foldseek_stats = extract_method_results(all_results, 'alignment_baselines', task, 'Foldseek', test_set, metric)
        if foldseek_stats:
            method_values['foldseek'] = foldseek_stats['mean']
        
        tm_align_stats = extract_method_results(all_results, 'alignment_baselines', task, 'TM-Align', test_set, metric)
        if tm_align_stats:
            method_values['tm_align'] = tm_align_stats['mean']
        
        return method_values
    
    # For backbone-specific methods
    # Get PLASMA value
    plasma_stats = extract_method_results(all_results, 'plasma', task, backbone, test_set, metric)
    if plasma_stats:
        method_values['plasma'] = plasma_stats['mean']
    
    # Get PLASMA-PF value
    plasma_pf_stats = extract_method_results(all_results, 'plasma_pf', task, backbone, test_set, metric)
    if plasma_pf_stats:
        method_values['plasma_pf'] = plasma_pf_stats['mean']
    
    # Get EBA value
    eba_stats = extract_method_results(all_results, 'eba', task, backbone, test_set, metric)
    if not eba_stats:
        eba_stats = extract_method_results(all_results, 'plasma_pf', task, backbone, test_set, metric)
    if eba_stats:
        method_values['eba'] = eba_stats['mean']
    
    # Get Backbone value
    backbone_stats = extract_method_results(all_results, 'baselines', task, backbone, test_set, metric)
    if backbone_stats:
        method_values['backbone'] = backbone_stats['mean']
    
    # Always include TM-Align baseline for gradient-based coloring comparison
    tm_align_stats = extract_method_results(all_results, 'alignment_baselines', task, 'TM-Align', test_set, metric)
    if tm_align_stats:
        method_values['tm_align'] = tm_align_stats['mean']
    
    # Also include Foldseek for context
    foldseek_stats = extract_method_results(all_results, 'alignment_baselines', task, 'Foldseek', test_set, metric)
    if foldseek_stats:
        method_values['foldseek'] = foldseek_stats['mean']
    
    return method_values


def standardize_backbone_names(results: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize backbone model names across different methods.
    
    Args:
        results: Results dictionary with potentially inconsistent naming
        
    Returns:
        Results dictionary with standardized names
    """
    # Mapping from various naming conventions to standardized names
    name_mapping = {
        'ankh': 'Ankh',
        'ankh-base': 'Ankh', 
        'esm2': 'ESM2',
        'esm2_t33_650M_UR50D': 'ESM2',
        'protbert': 'ProtBERT',
        'prot_bert': 'ProtBERT',
        'prott5': 'ProtT5',
        'prot_t5_xl_half_uniref50-enc': 'ProtT5',
        'prostt5': 'ProstT5',
        'ProstT5': 'ProstT5',
        'tm_vec': 'TM-Vec',
        'TM-Vec': 'TM-Vec',
        'protssn': 'ProtSSN',
        'ProtSSN': 'ProtSSN',
        'foldseek': 'Foldseek',
        'tm_align': 'TM-Align'
    }
    
    standardized = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for task, task_data in results.items():
        for method, method_data in task_data.items():
            # Standardize method name
            std_method = name_mapping.get(method, method)
            
            for split_or_test, data in method_data.items():
                standardized[task][std_method][split_or_test] = data
    
    return dict(standardized)


def load_all_results(config: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    """Load all results from different methods based on configuration.
    
    Args:
        config: Dictionary mapping method names to directory lists
        
    Returns:
        Dictionary mapping method names to their results
    """
    all_results = {}
    
    # Load PLASMA results
    if 'plasma' in config:
        plasma_results = {}
        for directory in config['plasma']:
            plasma_data = load_plasma_results(directory)
            # Merge results
            for task, task_data in plasma_data.items():
                if task not in plasma_results:
                    plasma_results[task] = {}
                for backbone, backbone_data in task_data.items():
                    if backbone not in plasma_results[task]:
                        plasma_results[task][backbone] = {}
                    plasma_results[task][backbone].update(backbone_data)
        
        all_results['plasma'] = standardize_backbone_names(plasma_results)
    
    # Load PLASMA-PF results
    if 'plasma_pf' in config:
        plasma_pf_results = {}
        for directory in config['plasma_pf']:
            plasma_pf_data = load_plasma_pf_results(directory)
            # Merge results
            for task, task_data in plasma_pf_data.items():
                if task not in plasma_pf_results:
                    plasma_pf_results[task] = {}
                for backbone, backbone_data in task_data.items():
                    if backbone not in plasma_pf_results[task]:
                        plasma_pf_results[task][backbone] = {}
                    plasma_pf_results[task][backbone].update(backbone_data)
        
        all_results['plasma_pf'] = standardize_backbone_names(plasma_pf_results)
    
    # Load EBA results
    if 'eba' in config:
        eba_results = {}
        for directory in config['eba']:
            eba_data = load_eba_results(directory)
            # Merge results
            for task, task_data in eba_data.items():
                if task not in eba_results:
                    eba_results[task] = {}
                for backbone, backbone_data in task_data.items():
                    if backbone not in eba_results[task]:
                        eba_results[task][backbone] = {}
                    eba_results[task][backbone].update(backbone_data)
        
        all_results['eba'] = standardize_backbone_names(eba_results)
    
    # Load baseline results
    if 'baselines' in config:
        baseline_results = {}
        for directory in config['baselines']:
            baseline_data = load_baseline_results(directory)
            # Merge results
            for task, task_data in baseline_data.items():
                if task not in baseline_results:
                    baseline_results[task] = {}
                for backbone, backbone_data in task_data.items():
                    if backbone not in baseline_results[task]:
                        baseline_results[task][backbone] = {}
                    baseline_results[task][backbone].update(backbone_data)
        
        all_results['baselines'] = standardize_backbone_names(baseline_results)
    
    # Load alignment baseline results
    if 'alignment_baselines' in config:
        align_results = {}
        for directory in config['alignment_baselines']:
            align_data = load_alignment_baseline_results(directory)
            # Merge results
            for task, task_data in align_data.items():
                if task not in align_results:
                    align_results[task] = {}
                for method, method_data in task_data.items():
                    if method not in align_results[task]:
                        align_results[task][method] = {}
                    align_results[task][method].update(method_data)
        
        all_results['alignment_baselines'] = standardize_backbone_names(align_results)
    
    return all_results


def extract_method_results(all_results: Dict[str, Dict[str, Any]], 
                         method_name: str, 
                         task: str, 
                         model: str, 
                         test_set: str, 
                         metric: str) -> Optional[Dict[str, Any]]:
    """Extract results for a specific method, task, model, test set, and metric.
    
    Args:
        all_results: All loaded results
        method_name: Method name (plasma, plasma_pf, baselines, alignment_baselines)
        task: Task name (active_site, binding_site, motif)
        model: Model/backbone name
        test_set: Test set name (test_frequent, test_hard)
        metric: Metric name (rocauc, pr_auc, f1_max, label_match_score)
        
    Returns:
        Dictionary with statistics or None if not found
    """
    try:
        method_results = all_results.get(method_name, {})
        task_results = method_results.get(task, {})
        model_results = task_results.get(model, {})
        
        # Collect values across splits
        values = []
        for split_data in model_results.values():
            if test_set in split_data and metric in split_data[test_set]:
                values.append(split_data[test_set][metric])
        
        if values:
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            se = std / np.sqrt(len(values)) if len(values) > 0 else 0.0
            
            return {
                'mean': mean,
                'std': std,
                'se': se,
                'n': len(values),
                'values': values
            }
        
        return None
        
    except KeyError:
        return None