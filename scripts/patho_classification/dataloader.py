import os
import numpy as np
import torch
from torch.utils.data import Dataset

class DiseaseDataset(Dataset):
    def __init__(self, data_path, dataset_type = 'train'):
        
        """
        初始化数据集，加载对应的索引和其他数据。

        参数:
        - data_path: 数据文件的路径
        - dataset_type: 数据集类型，可以是 'train', 'val', 'test'
        """
        
        
        self.data_path = data_path
        self.index = np.load(os.path.join(self.data_path, f'{dataset_type}_index.npy'))
        self.ref_emb = np.load(os.path.join(self.data_path, 'remb.npy'))
        self.mut_emb = np.load(os.path.join(self.data_path,'memb.npy'))
        self.ref_res = np.load(os.path.join(self.data_path,'rres.npy'))
        self.mut_res = np.load(os.path.join(self.data_path,'mres.npy'))
        self.y_class = np.load(os.path.join(self.data_path,'y_2.npy'))
        self.llm = np.load(os.path.join(self.data_path,'llm.npy'))
        self.other = np.load(os.path.join(self.data_path,'other.npy'))

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
        other = torch.from_numpy(self.other[index])
        
        return ref_emb, mut_emb, y_class, llm, other