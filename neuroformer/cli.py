"""
NeuroFormer Command-Line Interface.

Provides commands for training, evaluation, and inference.
"""

import argparse
import sys
import os
from pathlib import Path


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="NeuroFormer: State-of-the-art EEG Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neuroformer train --data train.csv --epochs 100
  neuroformer predict --model best_model.pth --input test.csv
  neuroformer evaluate --model best_model.pth --data test.csv
  neuroformer pretrain --data unlabeled.csv --method contrastive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train NeuroFormer model')
    train_parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    train_parser.add_argument('--val-data', type=str, help='Path to validation data CSV')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--output', type=str, default='./checkpoints', help='Output directory')
    train_parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    train_parser.add_argument('--pretrained', type=str, help='Path to pretrained weights')
    
    # Pretrain command
    pretrain_parser = subparsers.add_parser('pretrain', help='Self-supervised pretraining')
    pretrain_parser.add_argument('--data', type=str, required=True, help='Path to unlabeled data')
    pretrain_parser.add_argument('--method', type=str, choices=['contrastive', 'masked', 'both'],
                                 default='contrastive', help='Pretraining method')
    pretrain_parser.add_argument('--epochs', type=int, default=50, help='Pretraining epochs')
    pretrain_parser.add_argument('--output', type=str, default='./pretrained', help='Output directory')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    predict_parser.add_argument('--input', type=str, required=True, help='Input data file')
    predict_parser.add_argument('--output', type=str, help='Output predictions file')
    predict_parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--data', type=str, required=True, help='Path to test data')
    eval_parser.add_argument('--output', type=str, help='Output metrics file')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--model', type=str, help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'train':
        run_training(args)
    elif args.command == 'pretrain':
        run_pretraining(args)
    elif args.command == 'predict':
        run_prediction(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    elif args.command == 'info':
        show_info(args)


def run_training(args):
    """Run model training."""
    import torch
    from neuroformer.models import NeuroFormer
    from neuroformer.data import EEGDataset, EEGDataModule
    from neuroformer.training import Trainer, CombinedLoss
    from neuroformer.config import NeuroFormerConfig
    
    print("=" * 50)
    print("NeuroFormer Training")
    print("=" * 50)
    
    config = NeuroFormerConfig()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Device: {device}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # Load data
    dataset = EEGDataset.from_csv(args.data)
    
    # Create data module (will split if no val data provided)
    if args.val_data:
        val_dataset = EEGDataset.from_csv(args.val_data)
        from torch.utils.data import DataLoader
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    else:
        data_module = EEGDataModule(
            dataset.features.numpy(),
            dataset.labels.numpy(),
            batch_size=args.batch_size
        )
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
    
    # Create model
    model = NeuroFormer(
        num_electrodes=config.num_electrodes,
        num_classes=config.num_classes,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_transformer_layers=config.n_layers,
        dropout=config.dropout
    )
    
    # Load pretrained weights if provided
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded pretrained weights from {args.pretrained}")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.weight_decay)
    criterion = CombinedLoss(num_classes=config.num_classes)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=args.output
    )
    
    # Train
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs)
    
    print(f"\n✓ Training complete!")
    print(f"  Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"  Checkpoints saved to: {args.output}")


def run_pretraining(args):
    """Run self-supervised pretraining."""
    print("=" * 50)
    print("NeuroFormer Self-Supervised Pretraining")
    print("=" * 50)
    print(f"Method: {args.method}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    
    # Implementation would go here
    print("\n[Pretraining implementation - loads unlabeled data and runs SSL]")


def run_prediction(args):
    """Run inference on new data."""
    import torch
    import pandas as pd
    from neuroformer.models import NeuroFormer
    from neuroformer.inference import Predictor
    from neuroformer.data import EEGDataset
    
    print("=" * 50)
    print("NeuroFormer Prediction")
    print("=" * 50)
    
    # Load model
    predictor = Predictor.from_checkpoint(
        args.model,
        NeuroFormer,
        {'num_electrodes': 19, 'num_classes': 7}
    )
    
    # Load data
    dataset = EEGDataset.from_csv(args.input)
    
    # Predict
    results = predictor.get_top_k_predictions(dataset.features, k=args.top_k)
    
    print(f"\nPredictions for {len(results)} samples:")
    print("-" * 40)
    
    for i, preds in enumerate(results[:10]):  # Show first 10
        print(f"Sample {i+1}:")
        for rank, (name, prob) in enumerate(preds, 1):
            print(f"  {rank}. {name}: {prob:.3f}")
    
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more samples")
    
    # Save if output specified
    if args.output:
        predictions_df = pd.DataFrame({
            'sample_id': range(len(results)),
            'predicted_class': [r[0][0] for r in results],
            'confidence': [r[0][1] for r in results]
        })
        predictions_df.to_csv(args.output, index=False)
        print(f"\n✓ Predictions saved to {args.output}")


def run_evaluation(args):
    """Evaluate model on test data."""
    import torch
    from neuroformer.models import NeuroFormer
    from neuroformer.inference import Predictor
    from neuroformer.data import EEGDataset
    from neuroformer.training.metrics import compute_all_metrics
    from torch.utils.data import DataLoader
    
    print("=" * 50)
    print("NeuroFormer Evaluation")
    print("=" * 50)
    
    # Load model
    predictor = Predictor.from_checkpoint(
        args.model,
        NeuroFormer,
        {'num_electrodes': 19, 'num_classes': 7}
    )
    
    # Load data
    dataset = EEGDataset.from_csv(args.data)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # Get predictions
    results = predictor.predict_batch(dataloader)
    
    # Compute metrics
    metrics = compute_all_metrics(
        results['class_indices'],
        results['targets'],
        predictor.class_names
    )
    
    print("\nEvaluation Results:")
    print("-" * 40)
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"F1 (Macro):        {metrics['f1_macro']:.4f}")
    print(f"F1 (Weighted):     {metrics['f1_weighted']:.4f}")
    
    print("\nPer-class Results:")
    print("-" * 40)
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name}:")
        print(f"  Precision: {class_metrics['precision']:.3f}")
        print(f"  Recall:    {class_metrics['recall']:.3f}")
        print(f"  F1:        {class_metrics['f1']:.3f}")


def show_info(args):
    """Show model and package information."""
    from neuroformer import __version__
    from neuroformer.config import NeuroFormerConfig
    
    print("=" * 50)
    print("NeuroFormer Information")
    print("=" * 50)
    print(f"Version: {__version__}")
    print()
    
    config = NeuroFormerConfig()
    print("Default Configuration:")
    print(f"  Electrodes:  {config.num_electrodes}")
    print(f"  Classes:     {config.num_classes}")
    print(f"  Model dim:   {config.d_model}")
    print(f"  Heads:       {config.n_heads}")
    print(f"  Layers:      {config.n_layers}")
    
    if args.model:
        import torch
        checkpoint = torch.load(args.model, map_location='cpu')
        print(f"\nCheckpoint: {args.model}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best Val Acc: {checkpoint.get('best_val_acc', 'N/A'):.4f}")


if __name__ == '__main__':
    main()
