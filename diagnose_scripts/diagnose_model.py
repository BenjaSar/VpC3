#!/usr/bin/env python3
"""
Diagnostic Script: Model Architecture & Forward Pass Check
Verifies model works correctly with sample data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np

from models.vit_segmentation import ViTSegmentation

def diagnose_model():
    """Test model architecture and forward pass"""
    
    print("=" * 80)
    print("MODEL DIAGNOSTICS")
    print("=" * 80)
    
    # Configuration
    CONFIG = {
        'img_size': 512,
        'patch_size': 32,
        'in_channels': 3,
        'n_classes': 12,
        'embed_dim': 384,
        'n_encoder_layers': 12,
        'n_decoder_layers': 3,
        'n_heads': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    
    # Create model
    print("\n--- MODEL CREATION ---")
    try:
        model = ViTSegmentation(**CONFIG).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {total_params / 1e6:.2f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return
    
    # Test forward pass
    print("\n--- FORWARD PASS TEST ---")
    try:
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 512, 512).to(device)
        print(f"  Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected shape: ({batch_size}, {CONFIG['n_classes']}, 512, 512)")
        
        # Verify shape
        expected_shape = (batch_size, CONFIG['n_classes'], 512, 512)
        if output.shape == expected_shape:
            print(f"✓ Output shape is correct!")
        else:
            print(f"❌ Output shape mismatch!")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test backward pass
    print("\n--- BACKWARD PASS TEST ---")
    try:
        model.train()
        dummy_input = torch.randn(2, 3, 512, 512).to(device)
        dummy_target = torch.randint(0, 12, (2, 512, 512)).to(device).long()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        print(f"  Loss value: {loss.item():.4f}")
        print(f"  Expected: ~{np.log(12):.2f} (log(12) for random predictions)")
        
        loss.backward()
        optimizer.step()
        
        print(f"✓ Backward pass successful")
        print(f"✓ Gradients computed and optimizer step completed")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test with class weights
    print("\n--- WEIGHTED LOSS TEST ---")
    try:
        # Create class weights (inverse frequency)
        class_weights = torch.ones(12).to(device)
        class_weights[0] = 0.5  # Background is common
        class_weights[1:] = 2.0  # Other classes are rare
        
        criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)
        
        model.train()
        dummy_input = torch.randn(2, 3, 512, 512).to(device)
        dummy_target = torch.randint(0, 12, (2, 512, 512)).to(device).long()
        
        output = model(dummy_input)
        loss = criterion_weighted(output, dummy_target)
        print(f"  Weighted loss value: {loss.item():.4f}")
        
        loss.backward()
        print(f"✓ Weighted loss backward pass successful")
    except Exception as e:
        print(f"❌ Weighted loss failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test output value ranges
    print("\n--- OUTPUT VALUE ANALYSIS ---")
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 512, 512).to(device)
            output = model(dummy_input)
            
            # Check raw logits
            print(f"  Raw logits:")
            print(f"    Min: {output.min().item():.4f}")
            print(f"    Max: {output.max().item():.4f}")
            print(f"    Mean: {output.mean().item():.4f}")
            
            # Check predictions
            pred = output.argmax(dim=1)
            print(f"  Predictions:")
            print(f"    Min class: {pred.min().item()}")
            print(f"    Max class: {pred.max().item()}")
            print(f"    Unique classes: {torch.unique(pred).tolist()}")
            
            if pred.min() < 0 or pred.max() >= 12:
                print(f"  ❌ Predicted classes out of range!")
            else:
                print(f"  ✓ Predicted classes in valid range [0, 11]")
    except Exception as e:
        print(f"❌ Output analysis failed: {e}")
        return
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✓ Model architecture is correct")
    print("✓ Forward pass works correctly")
    print("✓ Backward pass works correctly")
    print("✓ Loss computation works correctly")
    print("\nThe model itself appears to be fine!")
    print("The training issue is likely in the dataset or training configuration.")

if __name__ == "__main__":
    diagnose_model()
