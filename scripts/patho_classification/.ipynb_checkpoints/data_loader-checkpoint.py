import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class DiseaseDataset(Dataset):
    def __init__(self, data_path):
        
        self.data_path = data_path
        self.index = np.load(os.path.join(self.data_path, 'index.npy'))
        self.ref_emb = np.load(os.path.join(self.data_path, 'remb.npy'))
        self.mut_emb = np.load(os.path.join(self.data_path,'memb.npy'))
        self.ref_res = np.load(os.path.join(self.data_path,'rres.npy'))
        self.mut_res = np.load(os.path.join(self.data_path,'mres.npy'))
        self.y_class = np.load(os.path.join(self.data_path,'y_2.npy'))
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
        ref_emb = torch.cat((emb1,res1),dim=1)
        mut_emb = torch.cat((emb2,res2),dim=1)
        llm = torch.from_numpy(self.llm[index])

        return ref_emb, mut_emb, y_class, llm

def load_data(data_dir, batch_size):
    
    y_class = {

    'pathogenic':            np.array([1., 0.]),
    'non-pathogenic':        np.array([0., 1.])  }

    class_indices = {key:i for i, key in enumerate(y_class.keys())}
    indices = [class_indices[key] for key in y_class.keys()]
    indices_tensor = torch.LongTensor(indices)
    
    dataset = DiseaseDataset(data_dir)  
    train_dataset, temp_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size = 0.5, random_state=42)
    

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_dataloader, val_dataloader, test_dataloader, indices_tensor
