#!/usr/bin/env python3
"""
Loss Matrix Analysis Script

This script analyzes three loss matrices by:
1. Loading the CSV files
2. Standardizing each row: (x_row_i - mean_i) / std_i
3. Computing the average loss curve across all rows
4. Plotting and saving the results

Usage:
    python analyze_loss_matrices.py [results_folder]
    
If no folder is provided, it defaults to 'run_results/2025-07-31_15-08-17/'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path


def load_matrix(filepath):
    """Load a CSV matrix file."""
    try:
        matrix = pd.read_csv(filepath, header=None)
        print(f"Loaded {filepath}: shape {matrix.shape}")
        return matrix.values
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def standardize_rows(matrix):
    """
    Standardize each row of the matrix: (x_row_i - mean_i) / std_i
    
    Args:
        matrix: numpy array of shape (n_rows, n_cols)
        
    Returns:
        standardized_matrix: numpy array of same shape with standardized rows
    """
    standardized = np.zeros_like(matrix, dtype=float)
    
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        row_mean = np.mean(row)
        row_std = np.std(row)
        
        # Avoid division by zero
        if row_std > 0:
            standardized[i, :] = (row - row_mean) / row_std
        else:
            # If std is 0, all values are the same, so standardized values are 0
            standardized[i, :] = 0
            
    return standardized


def compute_average_curve(standardized_matrix):
    """
    Compute the average loss curve across all rows.
    
    Args:
        standardized_matrix: numpy array with standardized rows
        
    Returns:
        average_curve: 1D numpy array with average values across rows
        std_curve: 1D numpy array with standard deviation across rows
    """
    average_curve = np.mean(standardized_matrix, axis=0)
    std_curve = np.std(standardized_matrix, axis=0)
    
    return average_curve, std_curve


def plot_results(results, output_dir):
    """
    Plot the average loss curves for all three matrices.
    
    Args:
        results: dict with matrix names as keys and (avg_curve, std_curve) as values
        output_dir: directory to save plots
    """
    plt.figure(figsize=(15, 10))
    
    colors = ['blue', 'red', 'green']
    matrix_names = ['Total KL', 'Test Reconstruction Error', 'Scalar Loss']
    
    for i, (key, (avg_curve, std_curve)) in enumerate(results.items()):
        epochs = np.arange(len(avg_curve))
        
        plt.subplot(2, 2, i + 1)
        plt.plot(epochs, avg_curve, color=colors[i], linewidth=2, label=f'{matrix_names[i]} (Mean)')
        plt.fill_between(epochs, avg_curve - std_curve, avg_curve + std_curve, 
                        color=colors[i], alpha=0.3, label=f'{matrix_names[i]} (±1 STD)')
        plt.xlabel('Epoch/Step')
        plt.ylabel('Standardized Value')
        plt.title(f'Average {matrix_names[i]} Curve (Standardized)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Combined plot
    plt.subplot(2, 2, 4)
    for i, (key, (avg_curve, std_curve)) in enumerate(results.items()):
        epochs = np.arange(len(avg_curve))
        plt.plot(epochs, avg_curve, color=colors[i], linewidth=2, label=matrix_names[i])
    
    plt.xlabel('Epoch/Step')
    plt.ylabel('Standardized Value')
    plt.title('All Average Loss Curves (Standardized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'standardized_loss_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.show()


def save_results(results, output_dir):
    """
    Save the standardized results to CSV files.
    
    Args:
        results: dict with matrix names as keys and (avg_curve, std_curve) as values
        output_dir: directory to save results
    """
    # Create a combined DataFrame with all results
    combined_data = {}
    
    for key, (avg_curve, std_curve) in results.items():
        matrix_name = key.replace('_matrix.csv', '').replace('_', ' ').title()
        combined_data[f'{matrix_name}_Mean'] = avg_curve
        combined_data[f'{matrix_name}_Std'] = std_curve
    
    # Add epoch/step column
    combined_data['Epoch'] = np.arange(len(list(results.values())[0][0]))
    
    # Reorder columns to put Epoch first
    columns = ['Epoch'] + [col for col in combined_data.keys() if col != 'Epoch']
    df = pd.DataFrame(combined_data)[columns]
    
    # Save to CSV
    results_path = os.path.join(output_dir, 'standardized_average_curves.csv')
    df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    return df


def analyze_matrices(results_folder):
    """
    Main analysis function.
    
    Args:
        results_folder: path to folder containing the matrix CSV files
    """
    # Define the three matrix files
    matrix_files = [
        'total_kl_matrix.csv',
        'test_reconstruction_error_matrix.csv', 
        'scalar_loss_matrix.csv'
    ]
    
    results = {}
    
    print(f"Analyzing matrices in: {results_folder}")
    print("=" * 60)
    
    for matrix_file in matrix_files:
        filepath = os.path.join(results_folder, matrix_file)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            continue
            
        # Load matrix
        matrix = load_matrix(filepath)
        if matrix is None:
            continue
            
        print(f"Matrix shape: {matrix.shape}")
        print(f"Original value range: [{np.min(matrix):.2f}, {np.max(matrix):.2f}]")
        
        # Standardize rows
        standardized_matrix = standardize_rows(matrix)
        print(f"Standardized value range: [{np.min(standardized_matrix):.2f}, {np.max(standardized_matrix):.2f}]")
        
        # Compute average curve
        avg_curve, std_curve = compute_average_curve(standardized_matrix)
        
        results[matrix_file] = (avg_curve, std_curve)
        
        print(f"Average curve computed with {len(avg_curve)} points")
        print(f"Mean of average curve: {np.mean(avg_curve):.4f}")
        print(f"Std of average curve: {np.std(avg_curve):.4f}")
        print("-" * 40)
    
    if not results:
        print("No matrices were successfully loaded!")
        return None
    
    # Create output directory
    output_dir = os.path.join(results_folder, 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot results
    plot_results(results, output_dir)
    
    # Save results
    df = save_results(results, output_dir)
    
    print("\nAnalysis Summary:")
    print("=" * 60)
    print(f"Processed {len(results)} matrices")
    print(f"Each matrix was standardized row-wise: (x - mean) / std")
    print(f"Average curves computed across all rows")
    print(f"Results saved to: {output_dir}")
    
    return results, df


def main():
    """Main function to run the analysis."""
    # Default folder
    # default_folder = 'run_results/2025-07-31_15-08-17/'
    
    # Get folder from command line argument or use default
    if len(sys.argv) > 1:
        results_folder = sys.argv[1]

    
    # Check if folder exists
    if not os.path.exists(results_folder):
        print(f"Error: Folder '{results_folder}' does not exist!")
        print(f"Please provide a valid folder path or ensure the default folder exists.")
        return
    
    # Run analysis
    try:
        results, df = analyze_matrices(results_folder)
        
        if results is not None:
            print("\nAnalysis completed successfully!")
            print("\nFirst few rows of combined results:")
            print(df.head(10))
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()