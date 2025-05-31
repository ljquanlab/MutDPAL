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
    'nervous system diseases':            np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    'digestive system diseases':          np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    'other congenital disorders':         np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    'congenital disorders of metabolism': np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    'reproductive system diseases':       np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    'cardiovascular diseases':            np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    'respiratory diseases':               np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
    'immune system diseases':             np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]),
    'endocrine and metabolic diseases':   np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
    'musculoskeletal diseases':           np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]),
    'urinary system diseases':            np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
    'skin diseases':                      np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]),
    'cancers':                            np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
    'Not provided':                       np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]),
    'unknown':                            np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
    # '-':                                  np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
}

    class_indices = {key:i for i, key in enumerate(y_class.keys())}
    indices = [class_indices[key] for key in y_class.keys()]
    indices_tensor = torch.LongTensor(indices)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    
    dataset = DiseaseDataset(args.data_dir)    
    train_indices, temp_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    # test_dataset = DiseaseDataset(args.data_dir, dataset_type='test')
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_indices), shuffle=False)

    # Initialize model
    model = MainModule(input_dim=args.input_size, num_heads=args.num_heads, ff_dim=args.ff_dim,
                       num_classes=args.num_classes, dis_emb=args.dis_emb)
    model.to(device)

    # Loss and optimizer
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train and evaluate
    if args.mode =='train':
        
        train(model, args.num_epochs, train_dataloader, val_dataloader, optimizer, loss_fn, indices_tensor, device,args.num_classes, args.save_path)
        
    elif args.mode =='test':
        
        test(model, test_dataloader, loss_fn, indices_tensor, args.save_path, device, args.num_classes, args.result_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default='../../data/features',
                        help="Path to the dataset directory")
    parser.add_argument("--input_size", type=int, default=1493, help="Input size of the model")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--ff_dim", type=int, default=256, help="Feedforward dimension")
    parser.add_argument("--dis_emb", type=int, default=128, help="Disease embedding dimension")
    parser.add_argument("--num_classes", type=int, default=15, help="Number of classes")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--save_path", type=str, default='../../models/multi_model/checkpoint.pth',
                        help="Path to save the model")
    parser.add_argument("--result_path", type=str, default='../../results/multi_result/result_m.txt',
                        help="Path to save the result")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help="Mode: 'train' or 'test'")
    args = parser.parse_args()

    main(args)
