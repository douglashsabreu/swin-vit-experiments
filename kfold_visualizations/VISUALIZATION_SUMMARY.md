# K-Fold Visualization Summary

## Generated Visualizations

### 1. kfold_accuracy_f1_comparison.png
- Comparison of Accuracy and F1-Score across all 5 folds
- Shows individual fold performance with mean lines
- Highlights consistency across folds

### 2. kfold_training_curves.png  
- Training curves for each individual fold
- Shows loss, accuracy, and F1-score progression
- Marks best epoch for each fold
- Includes summary statistics

### 3. kfold_statistical_analysis.png
- Box plots showing distribution of metrics
- Statistical summary table
- Training time analysis
- Coefficient of variation calculations

### 4. test_set_confusion_matrix.png
- Raw and normalized confusion matrices
- 750 unseen test files evaluation
- Shows per-class performance clearly

### 5. test_set_per_class_analysis.png
- Detailed per-class metrics (F1, Precision, Recall)
- Radar chart showing all metrics together
- Color-coded for easy interpretation

### 6. complete_thesis_summary.png
- Comprehensive overview of all results
- K-fold vs test set comparison
- Error analysis and confidence metrics
- Final summary statistics

## Key Findings

- **Excellent Generalization**: K-fold (95.25%) vs Test (95.20%)
- **Low Variance**: Std deviation < 1% across folds
- **Robust Methodology**: No data leakage, proper validation
- **Class-specific Insights**: Classes 510/514 most challenging
- **High Confidence**: Good model calibration

All visualizations are generated at 300 DPI for publication quality.
