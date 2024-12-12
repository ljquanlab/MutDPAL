import argparse
import torch
from torch import nn

def get_args():
    parser = argparse.ArgumentParser(description="MutDPAL pathogenic classification training and evaluation")

    # Dataset and paths
    parser.add_argument('--data_dir', type=str, default='../../data', help='Path to dataset')
    parser.add_argument('--save_path', type=str, default='../../model_state/train_best_2.pth', help='Path to save the model')
    parser.add_argument('--output_path', type=str, default='../../model_state/test_logs_2.txt', help='Path to save the model')

    # Hyperparameters
    
    parser.add_argument('--input_size', type=int, default=1280+213, help='Input size')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, default=256, help='Feedforward dimension')
    parser.add_argument('--dis_emb', type=int, default=128, help='Disease embedding size')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help="Mode: 'train' or 'test'")

    # Device
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    return args
