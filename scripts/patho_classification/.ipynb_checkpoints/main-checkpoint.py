import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataloader import DiseaseDataset
from model import MainModule
from train import train, test
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main(args):
    
    y_class = {
    'pathogenic':            np.array([1., 0.]),
    'non-pathogenic':        np.array([0., 1.])  }

    class_indices = {key:i for i, key in enumerate(y_class.keys())}
    indices = [class_indices[key] for key in y_class.keys()]
    indices_tensor = torch.LongTensor(indices)
    
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    train_dataset = DiseaseDataset(args.data_dir, dataset_type='train')
    val_dataset = DiseaseDataset(args.data_dir, dataset_type='val')
    test_dataset = DiseaseDataset(args.data_dir, dataset_type='test')
    

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = MainModule(input_dim=args.input_size, num_heads=args.num_heads, ff_dim=args.ff_dim,
                       num_classes=args.num_classes, dis_emb=args.dis_emb)
    model.to(device)

    # Loss and optimizer
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train and evaluate
    if args.mode =='train':
        train(model, args.num_epochs, train_dataloader, val_dataloader, optimizer, loss_fn, indices_tensor, device, args.save_path)
        
    elif args.mode =='test':
    
        test(model, test_dataloader, loss_fn, indices_tensor, args.save_path, device, args.result_path)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default='../../data/features',
                        help="Path to the dataset directory")
    parser.add_argument("--input_size", type=int, default=1493, help="Input size of the model")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--ff_dim", type=int, default=256, help="Feedforward dimension")
    parser.add_argument("--dis_emb", type=int, default=128, help="Disease embedding dimension")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")

    parser.add_argument("--save_path", type=str, default='../../results/patho_result/train_best_2.pth',
                        help="Path to save the model")
    parser.add_argument("--result_path", type=str, default='../../results/patho_result/result_2.txt',
                        help="Path to save the result")
    
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help="Mode: 'train' or 'test'")
    args = parser.parse_args()

    main(args)