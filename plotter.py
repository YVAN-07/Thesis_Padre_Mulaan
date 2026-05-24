#!/usr/bin/env python3
"""
PLOTTER FUNCTION REFERENCE - For importing and using plot functions

This module provides ready-to-use plotting functions for trial metrics.
Can be imported in other scripts or used as a standalone tool.

Example Usage:
    from plot_trial_metrics import plot_metric, find_all_csvs, read_csv_data
    
    # Get CSV files
    csv_files = find_all_csvs()
    
    # Process each file
    for csv_file in csv_files:
        data = read_csv_data(csv_file)
        if data:
            # Plot bidirectional progress
            plot_metric(data, 'bidirectional_norm', 
                       'Bidirectional Progress', 'my_plot.html')
            
            # Plot ground truth
            plot_metric(data, 'gt_norm',
                       'Ground Truth Distance', 'my_plot_gt.html')
"""

# ============================================================================
# AVAILABLE FUNCTIONS
# ============================================================================
#
# 1. find_trial_csvs(directory="controllers/so100_tele/logs")
#    └─ Finds per-trial CSV files (format: clip_bidir_so100_trial##_*.csv)
#
# 2. find_all_csvs(directory="controllers/so100_tele/logs")
#    └─ Finds all CSV files (per-trial and consolidated)
#
# 3. read_csv_data(filepath)
#    └─ Reads CSV and returns dict with columns:
#       - timestep, clip_similarity_raw, clip_similarity_ema
#       - bidirectional_raw, bidirectional_norm
#       - distance, gt_raw, gt_norm
#
# 4. plot_metric(csv_data, metric_name, metric_display_name, 
#               output_file, trial_number=None)
#    └─ Generates interactive HTML plot for a metric
#    └─ metric_name: 'bidirectional_norm' or 'gt_norm'
#    └─ output_file: path to save .html
#
# 5. compute_stats(data)
#    └─ Returns dict: {mean, min, max, count}
#
# 6. smooth_data(data, window=11)
#    └─ Applies moving average smoothing
#
# ============================================================================

# Import functions from main module
from plot_trial_metrics import (
    find_trial_csvs,
    find_all_csvs,
    read_csv_data,
    plot_metric,
    compute_stats,
    smooth_data,
    extract_trial_number,
    main
)

__all__ = [
    'find_trial_csvs',
    'find_all_csvs',
    'read_csv_data',
    'plot_metric',
    'compute_stats',
    'smooth_data',
    'extract_trial_number',
    'main'
]

if __name__ == "__main__":
    main()
