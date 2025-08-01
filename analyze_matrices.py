import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_analyze_matrices(results_dir):
    """Load and analyze the three matrices from the results directory."""
    
    # Define the matrix files
    matrix_files = {
        'Total KL': 'total_kl_matrix.csv',
        'Test Reconstruction Error': 'test_reconstruction_error_matrix.csv', 
        'Scalar Loss': 'scalar_loss_matrix.csv'
    }
    
    # Load the matrices
    matrices = {}
    for name, filename in matrix_files.items():
        filepath = Path(results_dir) / filename
        if filepath.exists():
            # Load without header since the data appears to be numeric only
            matrix = pd.read_csv(filepath, header=None)
            matrices[name] = matrix
            print(f"Loaded {name}: {matrix.shape}")
        else:
            print(f"Warning: {filepath} not found")
    
    return matrices

def plot_matrix_analysis(matrices, results_dir):
    """Create plots for each matrix showing individual rows and average."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with subplots for each matrix
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Matrix Analysis - {Path(results_dir).name}', fontsize=16, fontweight='bold')
    
    for idx, (name, matrix) in enumerate(matrices.items()):
        ax = axes[idx]
        
        # Plot each row as a thin line
        for row_idx in range(len(matrix)):
            row_data = matrix.iloc[row_idx].values
            ax.plot(row_data, alpha=0.3, linewidth=0.5, color='lightblue')
        
        # Calculate and plot the average row in bold
        avg_row = matrix.mean(axis=0)
        ax.plot(avg_row.values, color='red', linewidth=3, label='Average', alpha=0.8)
        
        # Customize the plot
        ax.set_title(f'{name}\n({matrix.shape[0]} rows × {matrix.shape[1]} cols)', 
                    fontweight='bold', fontsize=12)
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add some statistics as text
        stats_text = f'Avg Range: {avg_row.min():.1f} - {avg_row.max():.1f}\nStd: {avg_row.std():.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(results_dir) / 'matrix_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()
    
    return fig

def plot_individual_matrices(matrices, results_dir):
    """Create separate detailed plots for each matrix."""
    
    for name, matrix in matrices.items():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'{name} - Detailed Analysis', fontsize=16, fontweight='bold')
        
        # Top plot: All rows with average
        for row_idx in range(len(matrix)):
            row_data = matrix.iloc[row_idx].values
            ax1.plot(row_data, alpha=0.2, linewidth=0.5, color='skyblue')
        
        avg_row = matrix.mean(axis=0)
        ax1.plot(avg_row.values, color='red', linewidth=3, label='Average', alpha=0.9)
        ax1.set_title('All Rows + Average (Bold Red)')
        ax1.set_xlabel('Column Index')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Heatmap of the matrix
        im = ax2.imshow(matrix.values, aspect='auto', cmap='viridis', interpolation='nearest')
        ax2.set_title('Heatmap View')
        ax2.set_xlabel('Column Index')
        ax2.set_ylabel('Row Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Value')
        
        plt.tight_layout()
        
        # Save individual plot
        safe_name = name.replace(' ', '_').lower()
        output_path = Path(results_dir) / f'{safe_name}_detailed.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Detailed plot saved to: {output_path}")
        
        plt.show()

def print_summary_statistics(matrices):
    """Print summary statistics for each matrix."""
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for name, matrix in matrices.items():
        print(f"\n{name}:")
        print(f"  Shape: {matrix.shape}")
        print(f"  Overall mean: {matrix.values.mean():.2f}")
        print(f"  Overall std: {matrix.values.std():.2f}")
        print(f"  Min value: {matrix.values.min():.2f}")
        print(f"  Max value: {matrix.values.max():.2f}")
        
        # Average row statistics
        avg_row = matrix.mean(axis=0)
        print(f"  Average row mean: {avg_row.mean():.2f}")
        print(f"  Average row std: {avg_row.std():.2f}")
        print(f"  Average row range: {avg_row.min():.2f} to {avg_row.max():.2f}")

def main():
    # Set the results directory
    results_dir = "run_results/2025-07-31_15-08-17"
    
    print(f"Analyzing matrices in: {results_dir}")
    
    # Load the matrices
    matrices = load_and_analyze_matrices(results_dir)
    
    if not matrices:
        print("No matrices found to analyze!")
        return
    
    # Print summary statistics
    print_summary_statistics(matrices)
    
    # Create the main comparison plot
    plot_matrix_analysis(matrices, results_dir)
    
    # Create detailed individual plots
    plot_individual_matrices(matrices, results_dir)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()