import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DiseaseDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        
        self.index = np.load(os.path.join(self.data_path, 'index.npy'))
        
        self.ref_emb = np.load(os.path.join(self.data_path, 'ref.npy'))
        self.mut_emb = np.load(os.path.join(self.data_path,'mut.npy'))
        self.ref_res = np.load(os.path.join(self.data_path,'rres.npy'))
        self.mut_res = np.load(os.path.join(self.data_path,'mres.npy'))
        self.y_class = np.load(os.path.join(self.data_path,'y_class.npy'))
        self.llm = np.load(os.path.join(self.data_path,'llm.npy'))
            
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):

        index = self.index[idx]
        emb1 = torch.from_numpy(self.ref_emb[index])
        emb2 = torch.from_numpy(self.mut_emb[index])
        res1 = torch.from_numpy(self.ref_res[index])
        res2 = torch.from_numpy(self.mut_res[index])
        y_class = torch.from_numpy(self.y_class[index]) 
        llm = torch.from_numpy(self.llm[index])

        ref_emb = torch.cat((emb1,res1),dim = 1)
        mut_emb = torch.cat((emb2,res2),dim = 1)
        
        return ref_emb, mut_emb, y_class, llm 
    

    
def load_data(data_dir, batch_size):
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
    
    test_dataset = DiseaseDataset(data_dir)    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataloader, indices_tensor