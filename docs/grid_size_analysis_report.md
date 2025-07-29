# ARC Dataset Grid Size Distribution Analysis

## Overview

This report analyzes the distribution of grid sizes across all tasks in the ARC (Abstraction and Reasoning Corpus) dataset. The grid size is calculated as:

**Grid Size = n_examples × n_colors × n_x × n_y**

Where:
- `n_examples`: Number of input/output example pairs per task
- `n_colors`: Number of unique colors used in the task
- `n_x`: Maximum grid width across all examples
- `n_y`: Maximum grid height across all examples

## Dataset Summary

- **Total Tasks Analyzed**: 1,360
- **Training Split**: 1,000 tasks
- **Evaluation Split**: 120 tasks  
- **Test Split**: 240 tasks

## Key Findings

### 1. Split-wise Distribution Characteristics

The evaluation split contains significantly larger and more complex tasks compared to training and test splits:

| Split | Tasks | Mean Size | Median Size | Std Dev | Min | Max |
|-------|-------|-----------|-------------|---------|-----|-----|
| **Training** | 1,000 | 6,138 | 3,615 | 6,733 | 72 | 45,414 |
| **Evaluation** | 120 | 19,109 | 16,200 | 11,639 | 2,400 | 48,600 |
| **Test** | 240 | 6,360 | 4,154 | 6,347 | 108 | 32,400 |

### 2. Percentile Analysis

| Split | 25th | 75th | 90th | 95th |
|-------|------|------|------|------|
| **Training** | 1,620 | 8,100 | 15,216 | 20,187 |
| **Evaluation** | 11,104 | 25,380 | 36,450 | 40,511 |
| **Test** | 1,800 | 8,123 | 16,202 | 19,610 |

### 3. Size Category Distribution

Tasks categorized by grid size ranges:

| Category | Training | Evaluation | Test | Total |
|----------|----------|------------|------|-------|
| **Very Small (<1K)** | 151 (15.1%) | 0 (0.0%) | 31 (12.9%) | 182 (13.4%) |
| **Small (1K-5K)** | 442 (44.2%) | 10 (8.3%) | 106 (44.2%) | 558 (41.0%) |
| **Medium (5K-10K)** | 221 (22.1%) | 17 (14.2%) | 54 (22.5%) | 292 (21.5%) |
| **Large (10K-20K)** | 133 (13.3%) | 46 (38.3%) | 37 (15.4%) | 216 (15.9%) |
| **Very Large (>20K)** | 53 (5.3%) | 47 (39.2%) | 12 (5.0%) | 112 (8.2%) |

## Dimension Analysis

### Individual Dimension Ranges

| Split | Examples | Colors | X Dimension | Y Dimension |
|-------|----------|--------|-------------|-------------|
| **Training** | 3-12 | 1-9 | 2-30 | 1-30 |
| **Evaluation** | 3-8 | 1-9 | 7-30 | 10-30 |
| **Test** | 3-10 | 1-9 | 3-30 | 3-30 |

### Most Complex Tasks by Dimension

- **Most Examples**: `794b24be` (12 examples)
- **Most Colors**: `0a1d4ef5` (9 colors)
- **Largest Width**: `05a7bcf2` (30×30 grid)
- **Largest Height**: `05a7bcf2` (30×30 grid)

## Top 5 Largest Tasks

| Rank | Task ID | Split | Grid Size | Dimensions (examples×colors×x×y) |
|------|---------|-------|-----------|----------------------------------|
| 1 | `21897d95` | Evaluation | 48,600 | 6×9×30×30 |
| 2 | `8b7bacbf` | Evaluation | 48,600 | 6×9×30×30 |
| 3 | `88bcf3b4` | Evaluation | 45,927 | 7×9×27×27 |
| 4 | `4e7e0eb9` | Training | 45,414 | 6×9×29×29 |
| 5 | `e12f9a14` | Evaluation | 43,200 | 6×8×30×30 |

## Statistical Insights

### Overall Dataset Statistics
- **Overall Mean**: 7,322
- **Overall Median**: 4,320
- **Grid Size Range**: 72 - 48,600 (675× difference)
- **Distribution**: Highly right-skewed with long tail

### Key Observations

1. **Evaluation Complexity**: The evaluation split is systematically more complex:
   - 3× larger mean grid size than training/test
   - No tasks in the "Very Small" category
   - 77% of tasks are "Large" or "Very Large"

2. **Training vs Test Similarity**: Training and test splits have very similar distributions:
   - Nearly identical means (6,138 vs 6,360)
   - Similar category distributions
   - Comparable dimension ranges

3. **High Variability**: Grid sizes span nearly 3 orders of magnitude (72 to 48,600)

4. **Concentration**: Despite the wide range, 70% of all tasks have grid sizes under 10,000

## Visualizations

The analysis generated two comprehensive visualization files:

### 1. `plots/grid_size_distribution.png`
A 6-panel comprehensive analysis including:
- Grid size distribution by split
- Box plot comparison
- Grid dimensions scatter plot
- Color distribution
- Example count distribution
- Correlation matrix

### 2. `plots/grid_size_focused.png`
A 4-panel focused analysis including:
- Linear scale histogram
- Log scale histogram
- Box plot comparison
- Cumulative distribution function

## Data Files

- **`plots/grid_size_data.csv`**: Complete dataset with all task dimensions and calculated grid sizes
- Contains 1,360 rows with columns: split, task_name, n_examples, n_colors, n_x, n_y, grid_size, n_train, n_test

## Methodology

The analysis was conducted using the existing `preprocessing.py` module to ensure consistency with the project's task processing pipeline. Each task was processed to extract:

1. **Grid dimensions**: Maximum x and y dimensions across all examples
2. **Color count**: Number of unique colors (excluding background)
3. **Example count**: Total training + test examples per task
4. **Grid size calculation**: Product of all four dimensions

## Implications

This analysis reveals important characteristics of the ARC dataset:

1. **Evaluation Difficulty**: The evaluation split is designed to be significantly more challenging, with larger and more complex tasks
2. **Training Representativeness**: Training tasks cover a broad range of complexities but may under-represent the largest task sizes
3. **Computational Considerations**: Grid sizes vary dramatically, suggesting the need for adaptive computational strategies
4. **Memory Requirements**: The largest tasks require handling tensors with nearly 50,000 elements

---

*Analysis completed on ARC dataset containing 1,360 tasks across training, evaluation, and test splits.* 