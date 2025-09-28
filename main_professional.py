"""Professional Main Script for PhD Thesis Experiments."""

import argparse
import sys
from pathlib import Path

# Add project modules to path
sys.path.append(str(Path(__file__).parent))

from training.kfold_trainer import ProfessionalKFoldTrainer
from evaluation.detailed_test_evaluator import DetailedTestEvaluator
from tools.rigorous_data_split import RigorousDataSplitter
from src.config import load_config


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Professional PhD Thesis Experiments Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run rigorous data splitting
  python main_professional.py split --data-dir spatial_images_dataset_final
  
  # Run K-fold cross-validation
  python main_professional.py kfold --config experiments/spatial_experiment.yaml --folds 5
  
  # Run detailed test evaluation
  python main_professional.py evaluate --config experiments/spatial_experiment.yaml --checkpoint logs/best_checkpoint_52.pt
  
  # Run complete pipeline
  python main_professional.py pipeline --config experiments/spatial_experiment.yaml --data-dir spatial_images_dataset_final
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data splitting command
    split_parser = subparsers.add_parser('split', help='Rigorous data splitting')
    split_parser.add_argument('--data-dir', required=True, help='Input data directory')
    split_parser.add_argument('--output-dir', default='rigorous_splits', help='Output directory')
    split_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # K-fold training command
    kfold_parser = subparsers.add_parser('kfold', help='K-fold cross-validation training')
    kfold_parser.add_argument('--config', required=True, help='Configuration file path')
    kfold_parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    kfold_parser.add_argument('--output-dir', default='kfold_results', help='Output directory')
    
    # Test evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Detailed test evaluation')
    eval_parser.add_argument('--config', required=True, help='Configuration file path')
    eval_parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    eval_parser.add_argument('--output-dir', default='detailed_evaluation', help='Output directory')
    
    # Complete pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--config', required=True, help='Configuration file path')
    pipeline_parser.add_argument('--data-dir', help='Input data directory (if splitting needed)')
    pipeline_parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    pipeline_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser


def run_data_splitting(args) -> None:
    """Run rigorous data splitting."""
    print("🎓 Starting Rigorous Data Splitting")
    print("=" * 50)
    
    splitter = RigorousDataSplitter(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    report = splitter.run_complete_split()
    
    print("✅ Data splitting completed successfully!")
    print(f"📊 Split report saved to: {args.output_dir}")


def run_kfold_training(args) -> None:
    """Run K-fold cross-validation training."""
    print("🎓 Starting K-Fold Cross-Validation Training")
    print("=" * 50)
    
    config = load_config(args.config)
    
    trainer = ProfessionalKFoldTrainer(
        config=config,
        n_folds=args.folds,
        output_dir=args.output_dir
    )
    
    results = trainer.run_kfold_cv()
    
    print("✅ K-fold training completed successfully!")
    print(f"📊 Mean Accuracy: {results['mean_val_accuracy']:.4f} ± {results['std_val_accuracy']:.4f}")
    print(f"📊 Results saved to: {args.output_dir}")


def run_detailed_evaluation(args) -> None:
    """Run detailed test evaluation."""
    print("🎓 Starting Detailed Test Evaluation")
    print("=" * 50)
    
    evaluator = DetailedTestEvaluator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    results = evaluator.run_complete_evaluation()
    
    print("✅ Detailed evaluation completed successfully!")
    print(f"📊 Test Accuracy: {results['overall_metrics']['accuracy']:.4f}")
    print(f"📊 Test F1-Score: {results['overall_metrics']['macro_f1']:.4f}")
    print(f"📊 Results saved to: {args.output_dir}")


def run_complete_pipeline(args) -> None:
    """Run complete professional pipeline."""
    print("🎓 Starting Complete PhD Thesis Pipeline")
    print("=" * 70)
    
    # Step 1: Data splitting (if needed)
    if args.data_dir:
        print("\n📊 Step 1: Rigorous Data Splitting")
        print("-" * 40)
        
        splitter = RigorousDataSplitter(
            data_dir=args.data_dir,
            output_dir="rigorous_splits",
            seed=args.seed
        )
        splitter.run_complete_split()
        print("✅ Data splitting completed")
    
    # Step 2: K-fold cross-validation
    print("\n🔄 Step 2: K-Fold Cross-Validation")
    print("-" * 40)
    
    config = load_config(args.config)
    
    trainer = ProfessionalKFoldTrainer(
        config=config,
        n_folds=args.folds,
        output_dir="kfold_results"
    )
    
    kfold_results = trainer.run_kfold_cv()
    print(f"✅ K-fold completed - Mean Accuracy: {kfold_results['mean_val_accuracy']:.4f}")
    
    # Step 3: Find best checkpoint and evaluate
    print("\n🔍 Step 3: Detailed Test Evaluation")
    print("-" * 40)
    
    # Find best fold checkpoint
    kfold_dir = Path("kfold_results")
    best_checkpoint = None
    best_accuracy = 0
    
    for i in range(args.folds):
        checkpoint_path = kfold_dir / f"fold_{i}_best.pt"
        if checkpoint_path.exists():
            fold_results = kfold_results['individual_fold_results'][i]
            if fold_results['best_val_accuracy'] > best_accuracy:
                best_accuracy = fold_results['best_val_accuracy']
                best_checkpoint = checkpoint_path
    
    if best_checkpoint:
        evaluator = DetailedTestEvaluator(
            config_path=args.config,
            checkpoint_path=str(best_checkpoint),
            output_dir="detailed_evaluation"
        )
        
        eval_results = evaluator.run_complete_evaluation()
        print(f"✅ Evaluation completed - Test Accuracy: {eval_results['overall_metrics']['accuracy']:.4f}")
    else:
        print("⚠️ No valid checkpoints found for evaluation")
    
    print("\n" + "=" * 70)
    print("🎉 Complete PhD Pipeline Finished Successfully!")
    print("📊 All results are ready for thesis documentation")
    
    # Summary
    print("\n📋 Pipeline Summary:")
    if args.data_dir:
        print("   ✅ Data splitting: rigorous_splits/")
    print(f"   ✅ K-fold results: kfold_results/ (Mean Acc: {kfold_results['mean_val_accuracy']:.4f})")
    if best_checkpoint:
        print(f"   ✅ Test evaluation: detailed_evaluation/ (Test Acc: {eval_results['overall_metrics']['accuracy']:.4f})")
    print("   ✅ WandB logging: Automatic during training and evaluation")


def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'split':
            run_data_splitting(args)
        elif args.command == 'kfold':
            run_kfold_training(args)
        elif args.command == 'evaluate':
            run_detailed_evaluation(args)
        elif args.command == 'pipeline':
            run_complete_pipeline(args)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
