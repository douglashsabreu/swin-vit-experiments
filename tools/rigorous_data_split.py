"""Rigorous data splitting for PhD thesis with full traceability."""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict, Counter
import shutil
import pandas as pd

class RigorousDataSplitter:
    """PhD-grade data splitting with full traceability and documentation."""
    
    def __init__(self, data_dir: str, output_dir: str = "rigorous_splits", seed: int = 42):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split ratios
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        self.test_ratio = 0.1
        
        # Data tracking
        self.file_registry = {
            'train': [],
            'val': [],
            'test': []
        }
        
    def analyze_dataset(self) -> dict:
        """Comprehensive dataset analysis before splitting."""
        print("🔍 Analyzing dataset structure...")
        
        analysis = {
            'total_files': 0,
            'classes': {},
            'class_distribution': {},
            'file_extensions': defaultdict(int),
            'directory_structure': {}
        }
        
        all_files = []
        all_labels = []
        
        # Scan all class directories
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                class_name = class_dir.name
                class_files = list(class_dir.glob('*'))
                class_files = [f for f in class_files if f.is_file() and not f.name.startswith('.')]
                
                analysis['classes'][class_name] = {
                    'count': len(class_files),
                    'files': [str(f.relative_to(self.data_dir)) for f in class_files]
                }
                
                # Track files for splitting
                for file_path in class_files:
                    all_files.append(str(file_path.relative_to(self.data_dir)))
                    all_labels.append(class_name)
                    
                    # Track file extensions
                    analysis['file_extensions'][file_path.suffix.lower()] += 1
                
                analysis['total_files'] += len(class_files)
                analysis['class_distribution'][class_name] = len(class_files)
        
        # Calculate statistics
        analysis['class_balance'] = self._calculate_balance_metrics(analysis['class_distribution'])
        analysis['recommended_splits'] = self._calculate_split_sizes(analysis['total_files'])
        
        return analysis, all_files, all_labels
    
    def _calculate_balance_metrics(self, class_dist: dict) -> dict:
        """Calculate class balance metrics."""
        counts = list(class_dist.values())
        return {
            'min_class_size': min(counts),
            'max_class_size': max(counts),
            'mean_class_size': np.mean(counts),
            'std_class_size': np.std(counts),
            'balance_ratio': min(counts) / max(counts),
            'is_balanced': (min(counts) / max(counts)) > 0.8
        }
    
    def _calculate_split_sizes(self, total_files: int) -> dict:
        """Calculate exact split sizes."""
        return {
            'train': int(total_files * self.train_ratio),
            'val': int(total_files * self.val_ratio),
            'test': int(total_files * self.test_ratio),
            'total_allocated': int(total_files * (self.train_ratio + self.val_ratio + self.test_ratio))
        }
    
    def perform_stratified_split(self, all_files: list, all_labels: list) -> dict:
        """Perform stratified split with full documentation."""
        print("📊 Performing stratified split...")
        
        # Convert to arrays
        files_array = np.array(all_files)
        labels_array = np.array(all_labels)
        
        # First split: separate test set (10%)
        splitter1 = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=self.test_ratio, 
            random_state=self.seed
        )
        
        train_val_idx, test_idx = next(splitter1.split(files_array, labels_array))
        
        # Second split: separate train/val from remaining 90%
        remaining_val_ratio = self.val_ratio / (self.train_ratio + self.val_ratio)
        
        splitter2 = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=remaining_val_ratio, 
            random_state=self.seed
        )
        
        train_idx, val_idx = next(splitter2.split(files_array[train_val_idx], labels_array[train_val_idx]))
        
        # Map back to original indices
        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx]
        
        # Create split registry
        splits = {
            'train': {
                'files': files_array[train_idx].tolist(),
                'labels': labels_array[train_idx].tolist(),
                'indices': train_idx.tolist()
            },
            'val': {
                'files': files_array[val_idx].tolist(),
                'labels': labels_array[val_idx].tolist(),
                'indices': val_idx.tolist()
            },
            'test': {
                'files': files_array[test_idx].tolist(),
                'labels': labels_array[test_idx].tolist(),
                'indices': test_idx.tolist()
            }
        }
        
        return splits
    
    def validate_splits(self, splits: dict, analysis: dict) -> dict:
        """Validate split quality and balance."""
        print("✅ Validating split quality...")
        
        validation_report = {
            'split_sizes': {},
            'class_distributions': {},
            'balance_preservation': {},
            'quality_metrics': {}
        }
        
        original_dist = analysis['class_distribution']
        
        for split_name, split_data in splits.items():
            split_dist = Counter(split_data['labels'])
            validation_report['split_sizes'][split_name] = len(split_data['files'])
            validation_report['class_distributions'][split_name] = dict(split_dist)
            
            # Calculate preservation ratios
            preservation = {}
            for class_name in original_dist.keys():
                original_count = original_dist[class_name]
                split_count = split_dist.get(class_name, 0)
                preservation[class_name] = split_count / original_count if original_count > 0 else 0
            
            validation_report['balance_preservation'][split_name] = preservation
        
        # Quality metrics
        expected_ratios = {'train': self.train_ratio, 'val': self.val_ratio, 'test': self.test_ratio}
        total_files = sum(validation_report['split_sizes'].values())
        
        for split_name, expected_ratio in expected_ratios.items():
            actual_ratio = validation_report['split_sizes'][split_name] / total_files
            validation_report['quality_metrics'][f'{split_name}_ratio_error'] = abs(actual_ratio - expected_ratio)
        
        return validation_report
    
    def create_split_directories(self, splits: dict):
        """Create physical directory structure for splits."""
        print("📁 Creating split directories...")
        
        for split_name, split_data in splits.items():
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Create class subdirectories and copy files
            for file_path, label in zip(split_data['files'], split_data['labels']):
                class_dir = split_dir / label
                class_dir.mkdir(parents=True, exist_ok=True)
                
                source_path = self.data_dir / file_path
                dest_path = class_dir / Path(file_path).name
                
                if not dest_path.exists():
                    shutil.copy2(source_path, dest_path)
    
    def generate_comprehensive_report(self, analysis: dict, splits: dict, validation: dict) -> dict:
        """Generate comprehensive report for thesis documentation."""
        print("📋 Generating comprehensive report...")
        
        report = {
            'metadata': {
                'split_timestamp': pd.Timestamp.now().isoformat(),
                'random_seed': self.seed,
                'split_ratios': {
                    'train': self.train_ratio,
                    'val': self.val_ratio, 
                    'test': self.test_ratio
                },
                'splitting_method': 'stratified_shuffle_split',
                'data_source': str(self.data_dir)
            },
            'dataset_analysis': analysis,
            'split_details': splits,
            'validation_report': validation,
            'file_manifests': {}
        }
        
        # Create detailed file manifests
        for split_name, split_data in splits.items():
            manifest = []
            for i, (file_path, label) in enumerate(zip(split_data['files'], split_data['labels'])):
                manifest.append({
                    'index': i,
                    'file_path': file_path,
                    'label': label,
                    'original_index': split_data['indices'][i],
                    'split': split_name
                })
            
            report['file_manifests'][split_name] = manifest
        
        return report
    
    def save_documentation(self, report: dict):
        """Save all documentation for thesis traceability."""
        print("💾 Saving documentation...")
        
        # Save main report
        with open(self.output_dir / 'split_report.json', 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save individual manifests as CSV for easy viewing
        for split_name, manifest in report['file_manifests'].items():
            df = pd.DataFrame(manifest)
            df.to_csv(self.output_dir / f'{split_name}_manifest.csv', index=False)
        
        # Save summary statistics
        summary = {
            'total_files': report['dataset_analysis']['total_files'],
            'classes': list(report['dataset_analysis']['classes'].keys()),
            'split_sizes': report['validation_report']['split_sizes'],
            'class_distributions': report['validation_report']['class_distributions'],
            'quality_metrics': report['validation_report']['quality_metrics']
        }
        
        with open(self.output_dir / 'split_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create human-readable report
        self._create_readable_report(report)
    
    def _create_readable_report(self, report: dict):
        """Create human-readable markdown report."""
        report_md = f"""# PhD Thesis Dataset Split Report

## Metadata
- **Split Date**: {report['metadata']['split_timestamp']}
- **Random Seed**: {report['metadata']['random_seed']}
- **Splitting Method**: {report['metadata']['splitting_method']}
- **Data Source**: {report['metadata']['data_source']}

## Split Ratios
- **Training**: {report['metadata']['split_ratios']['train']:.1%}
- **Validation**: {report['metadata']['split_ratios']['val']:.1%}
- **Test**: {report['metadata']['split_ratios']['test']:.1%}

## Dataset Analysis
- **Total Files**: {report['dataset_analysis']['total_files']:,}
- **Number of Classes**: {len(report['dataset_analysis']['classes'])}
- **Classes**: {', '.join(report['dataset_analysis']['classes'].keys())}

### Class Distribution
"""
        
        for class_name, count in report['dataset_analysis']['class_distribution'].items():
            percentage = count / report['dataset_analysis']['total_files'] * 100
            report_md += f"- **{class_name}**: {count:,} files ({percentage:.1f}%)\n"
        
        report_md += f"""
### Balance Metrics
- **Balance Ratio**: {report['dataset_analysis']['class_balance']['balance_ratio']:.3f}
- **Is Balanced**: {report['dataset_analysis']['class_balance']['is_balanced']}
- **Min Class Size**: {report['dataset_analysis']['class_balance']['min_class_size']:,}
- **Max Class Size**: {report['dataset_analysis']['class_balance']['max_class_size']:,}

## Split Results
"""
        
        for split_name, size in report['validation_report']['split_sizes'].items():
            percentage = size / report['dataset_analysis']['total_files'] * 100
            report_md += f"- **{split_name.title()}**: {size:,} files ({percentage:.1f}%)\n"
        
        report_md += """
## Quality Validation
"""
        for metric_name, error in report['validation_report']['quality_metrics'].items():
            report_md += f"- **{metric_name.replace('_', ' ').title()}**: {error:.4f}\n"
        
        with open(self.output_dir / 'SPLIT_REPORT.md', 'w') as f:
            f.write(report_md)
    
    def run_complete_split(self) -> dict:
        """Execute complete rigorous splitting pipeline."""
        print("🎓 Starting Rigorous Data Splitting for PhD Thesis")
        print("=" * 60)
        
        # Step 1: Analyze dataset
        analysis, all_files, all_labels = self.analyze_dataset()
        
        print(f"📊 Dataset Overview:")
        print(f"   • Total files: {analysis['total_files']:,}")
        print(f"   • Classes: {len(analysis['classes'])}")
        print(f"   • Balance ratio: {analysis['class_balance']['balance_ratio']:.3f}")
        
        # Step 2: Perform split
        splits = self.perform_stratified_split(all_files, all_labels)
        
        # Step 3: Validate splits
        validation = self.validate_splits(splits, analysis)
        
        print(f"✅ Split completed:")
        for split_name, size in validation['split_sizes'].items():
            print(f"   • {split_name.title()}: {size:,} files")
        
        # Step 4: Create directories
        self.create_split_directories(splits)
        
        # Step 5: Generate report
        report = self.generate_comprehensive_report(analysis, splits, validation)
        
        # Step 6: Save documentation
        self.save_documentation(report)
        
        print("=" * 60)
        print("✅ Rigorous splitting completed!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📋 Check SPLIT_REPORT.md for detailed analysis")
        
        return report


def main():
    """Execute rigorous data splitting."""
    splitter = RigorousDataSplitter(
        data_dir="spatial_images_dataset_final",
        output_dir="rigorous_splits",
        seed=42
    )
    
    report = splitter.run_complete_split()
    return report

if __name__ == "__main__":
    main()
