import copy
from itertools import cycle
import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, hamming_loss, precision_score, recall_score, accuracy_score, coverage_error, auc, roc_auc_score, matthews_corrcoef, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler
from collections import Counter
import matplotlib.pyplot as plt
import warnings
from argparse import ArgumentParser
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')


"""
    初始化数据集，加载对应的索引和其他数据。

     参数:
     - data_path: 数据文件的路径
     - index: 数据索引
"""

class DiseaseDataset(Dataset):
    def __init__(self, data_path, index):
        
        self.data_path = data_path
        self.ref_emb = np.load(os.path.join(data_path, 'remb.npy'))
        self.mut_emb = np.load(os.path.join(data_path, 'memb.npy'))
        self.ref_res = np.load(os.path.join(data_path, 'rres.npy'))
        self.mut_res = np.load(os.path.join(data_path, 'mres.npy'))
        self.y_class = np.load(os.path.join(data_path, 'y_2.npy'))
        self.llm = np.load(os.path.join(data_path, 'llm.npy'))
        self.other = np.load(os.path.join(data_path, 'aaindex123pphak.npy'))
        
        self.index = index  # 直接使用传入的索引

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
    

"""  
MutDPAL  Module

"""
    
class ProtFeatModule(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim):
        super(ProtFeatModule, self).__init__()
        self.lstm1 = nn.LSTM(input_size= input_dim, hidden_size=512, num_layers=1, batch_first = True, bidirectional=True)
        self.linear1 = nn.Linear(512*2, 512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        sattn1 = nn.TransformerEncoderLayer(d_model = 128, nhead = num_heads, dim_feedforward = ff_dim, batch_first = True)
        self.encoder = nn.TransformerEncoder(sattn1, 4) 

    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        x = self.relu1(self.linear1(lstm_out))
        x = self.relu2(self.linear2(x))
        x = self.encoder(x)
        return x 


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, kdim, vdim, ff_dim, dropout = 0.1):
        super(CrossAttention, self).__init__()
        self.cattn = nn.MultiheadAttention(d_model, num_heads, kdim=kdim, vdim=vdim,
                                           dropout=dropout,batch_first=True)
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def _ca_block(self, q, k, v):
        x = self.cattn(q, k, v, need_weights = False)[0]
        return self.dropout1(x)
    def _ff_block(self, x):
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)
        
    def forward(self, q, k, v):
        x = q
        x = self.norm1(x + self._ca_block(q,k,v))
        x = self.norm2(x + self._ff_block(x))
        return x

    
class DiseaseEmbedding(nn.Module):
    def __init__(self, num_classes,embedding_dim):
        super(DiseaseEmbedding,self).__init__()
        self.embeddding = nn.Embedding(num_classes, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        x = self.embeddding(x)
        x = self.prelu(self.linear1(x))
        return x    


class LLMEmbedding(nn.Module):
    def __init__(self):
        super(LLMEmbedding, self).__init__()
        self.linear1 = nn.Linear(768, 256)
        self.conv1 = nn.Conv1d(768, 512, kernel_size = 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(512, 256, kernel_size = 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self,x1, x2):
        x1 = self.linear1(x1)
        x2 = x2.unsqueeze(-1)
        x2 = self.relu1(self.conv1(x2))
        x2 = self.relu2(self.conv2(x2))
        x2 = x2.squeeze(-1)

        x = torch.cat((x1, x2),dim=-1)
        return x


    
class MultiClassification(nn.Module):
    def __init__(self):
        super(MultiClassification, self).__init__()
        
        # self.fc2 = nn.Linear(512 + 256 + 256+3 , 128)
        self.fc2 = nn.Linear(256*2 +512 + 16, 128)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)  # 添加dropout层，丢弃概率为0.5
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x

    
class MainModule(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_classes, dis_emb):
        super(MainModule, self).__init__()
        self.prot = ProtFeatModule(input_dim=input_dim, num_heads=num_heads, ff_dim=ff_dim)
        self.disease = DiseaseEmbedding(num_classes=num_classes, embedding_dim=dis_emb)
        self.emb1 = CrossAttention(d_model=dis_emb, num_heads=num_heads, kdim=128, vdim=128, ff_dim=ff_dim) 
        self.emb2 = CrossAttention(d_model=dis_emb, num_heads=num_heads, kdim=128, vdim=128, ff_dim=ff_dim) 
        self.llm = LLMEmbedding()
        self.fc = nn.Linear(537, 16)
        self.classifier = MultiClassification()
        
    def forward(self, ref_prot, mut_prot, indices_tensor, llm, other):
        ref = self.prot(ref_prot)
        mut = self.prot(mut_prot) 
        dis_emb = self.disease(indices_tensor)
        ref_emb = self.emb1(dis_emb, ref, ref)
        mut_emb = self.emb1(dis_emb, mut, mut)
        diff1 = ref_emb - mut_emb
        dis_prot = self.emb2(ref_emb, mut_emb, mut_emb)
        output = torch.cat((dis_prot, diff1), dim=2)
        output = output.view(output.size(0), 2*256)
        other = self.fc(other)
        llm_emb = self.llm(llm,llm)  # 256
        concat_emb = torch.cat((output, llm_emb, other),dim=1)
        return concat_emb



def train(model, num_epochs, train_dataloader, val_dataloader, optimizer, loss_fn, indices_tensor, device, save_path, output_file):         
    
    #Begin Training
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_truth, train_preds, train_labels = [], [], []
        for ref_emb, mut_emb, y_class, llm, other in tqdm(train_dataloader):
            optimizer.zero_grad()
            ref_emb, mut_emb, y_class, llm, other = ref_emb.to(device), mut_emb.to(device),y_class.to(device),llm.to(device), other.to(device)

            current_batch_size = ref_emb.size(0)
            indices_tensor = indices_tensor.to(device)
            indices_tensor_disease = indices_tensor.repeat(current_batch_size, 1)

            output_emb = model(ref_emb, mut_emb, indices_tensor_disease,llm, other)
            output = model.classifier(output_emb)

            loss = loss_fn(output, y_class)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss/len(train_dataloader)

    #Begin Testing
    model.eval()
    val_loss = 0.0

    val_truth, val_preds, val_labels = [], [], []
    metrics_dict = {'recall':[],'precision':[],'accuracy':[],'auc':[],'sp':[],'mcc':[], 
           'ap':[]}

    with torch.no_grad():
        for ref_emb, mut_emb, y_class,llm, other in val_dataloader:
            ref_emb, mut_emb, y_class, llm, other = ref_emb.to(device), mut_emb.to(device),  y_class.to(device), llm.to(device), other.to(device)
            current_batch_size = ref_emb.size(0)
            indices_tensor = indices_tensor.to(device)
            indices_tensor_disease = indices_tensor.repeat(current_batch_size, 1)

            output_emb = model(ref_emb, mut_emb, indices_tensor_disease,llm, other)
            val_output = model.classifier(output_emb)

            val_label = (val_output.cpu().detach().numpy() > 0.5).astype(int)

            loss = loss_fn(val_output, val_label).mean()
            val_loss += loss.item()

            val_truth.append(y_class.detach().cpu().numpy())
            val_preds.append(val_output.detach().cpu().numpy())
            val_labels.append(val_label)

    y_classes = np.concatenate(val_truth, axis=0)
    y_scores = np.concatenate(val_preds, axis=0)
    y_labels = np.concatenate(val_labels, axis=0)

    val_loss /= len(val_dataloader)

    acc = np.mean(y_labels == y_classes)
    auc = roc_auc_score(y_classes, y_scores)
    ap = average_precision_score(y_classes, y_scores)
    mcc = matthews_corrcoef(y_classes, y_labels)
    recall = recall_score(y_classes, y_labels)
    precision = precision_score(y_classes, y_labels)
    f1 = f1_score(y_classes, y_labels)

    tn, fp, fn, tp = confusion_matrix(y_classes, y_labels).ravel()
    pn = tn+fp
    specificity = tn / pn

    with open(output_file, "a") as f:
        f.write('**********\n')
        f.write(f'Test Loss: {val_loss:.6f}, AUC: {auc:.6f}, ACC: {acc:.6f}, MCC: {mcc:.6f}, Recall: {recall:.6f}, Precision: {precision:.6f}, Sp: {specificity:.6f}, F1: {f1:.6f}\n')
        f.write('**********\n')

            
def test(model, test_dataloader, loss_fn, indices_tensor, save_path, device, output_file):
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    model.eval()
    test_loss = 0        
    test_truth, test_preds, test_labels = [], [], []
    all_output_embs = []
    
    with torch.no_grad():
        for ref_emb, mut_emb, y_class,llm, other in test_dataloader:
            ref_emb, mut_emb, y_class,llm, other = ref_emb.to(device), mut_emb.to(device), y_class.to(device), llm.to(device), other.to(device)
            
            current_batch_size = ref_emb.size(0)
            indices_tensor = indices_tensor.to(device)
            indices_tensor_disease = indices_tensor.repeat(current_batch_size, 1)
            
            output_emb = model(ref_emb, mut_emb, indices_tensor_disease,llm, other)
            output = model.classifier(output_emb)
            label = (output.cpu().detach().numpy() > 0.5).astype(int)
            
            loss = loss_fn(output, y_class)
            test_loss += loss.item()
            
            test_truth.append(y_class.cpu().numpy())
            test_preds.append(output.cpu().numpy())
            test_labels.append(label)

    y_pred_total = np.concatenate(test_preds, axis=0)
    y_true_total = np.concatenate(test_truth, axis=0)
    y_labels_total = np.concatenate(test_labels, axis=0)
    
    test_loss /= len(test_dataloader)

    acc = np.mean(y_labels_total == y_true_total)
    auc = roc_auc_score(y_true_total, y_pred_total)
    ap = average_precision_score(y_true_total, y_pred_total)
    mcc = matthews_corrcoef(y_true_total, y_labels_total)
    recall = recall_score(y_true_total, y_labels_total)
    precision = precision_score(y_true_total, y_labels_total)
    f1 = f1_score(y_true_total, y_labels_total)

    tn, fp, fn, tp = confusion_matrix(y_true_total, y_labels_total).ravel()
    pn = tn+fp
    specificity = tn / pn

    with open(output_file, "a") as f: 
        f.write(f'test Loss : {loss:.6f}, ACC: {acc:.6f}, AUC  : {auc:.6f},MCC: {mcc:.6f},Recall  : {recall:.6f},Precision  : {precision:.6f},Sp: {specificity:.6f}, F1:{f1:.6f} \n')

        
if __name__ == '__main__':
    
    parser = ArgumentParser(description="Train or test the model")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help="Mode: 'train' or 'test'")
    args = parser.parse_args()
    
    y_class = {
    'pathogenic':            np.array([1., 0.]),
    'non-pathogenic':        np.array([0., 1.])  }

    class_indices = {key:i for i, key in enumerate(y_class.keys())}
    indices = [class_indices[key] for key in y_class.keys()]
    indices_tensor = torch.LongTensor(indices)
    
    #设置超参数
    input_size = 1280 + 213
    num_heads = 4
    ff_dim = 256
    dis_emb = 128
    num_classes = 2
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 128
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_path= '../../data/pred-features'
    fold_path = '../../data/pred-features/10-fold_index'
    result_dir = '../../results/pred-muthtp/log'
    model_dir = '../../models/pred-muthtp/log'


    fold_indices_files = [f for f in os.listdir(fold_path) if f.startswith("fold_") and f.endswith("_index.npy")]
    fold_indices_files.sort()

    #十折交叉验证
    if args.mode == 'train':
        for fold in range(10):
            print(f"\n=== 训练第 {fold + 1} 折 ===")

            val_fold_file = fold_indices_files[fold]
            val_index = np.load(os.path.join(fold_path, val_fold_file))

            train_indices = []
            for i, file in enumerate(fold_indices_files):
                if i != fold:
                    train_indices.extend(np.load(os.path.join(fold_path, file)))

            train_index = np.array(train_indices, dtype=np.int32)


            train_dataset = DiseaseDataset(data_path, train_index)
            val_dataset = DiseaseDataset(data_path, val_index)

            # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = MainModule(input_size, num_heads, ff_dim, num_classes, dis_emb)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            loss_fn = torch.nn.BCELoss()

            result_path = os.path.join(result_dir, str(fold))
            model_dirs = os.path.join(model_dir, str(fold))
            os.makedirs(os.path.join(result_path, "log"), exist_ok=True)  
            os.makedirs(os.path.join(model_dirs, "log"), exist_ok=True)  

            result_txt = os.path.join(result_path, 'log', 'test_result.txt')
            model_path = os.path.join(model_dirs, 'log', 'checkpoint.pth')

            train_state, train_loss = train(model, num_epochs, train_loader, val_loader, optimizer,
                                        loss_fn, indices_tensor, device, model_path, result_txt)
        
            
            print(f"=== 第 {fold + 1} 折训练结束 ===\n")

            del model
            torch.cuda.empty_cache()

            del train_dataset, val_dataset, val_loader
            gc.collect()

        print("十折交叉验证完成！")
        
    elif args.mode == 'test':
        #独立测试
        test_index = np.load(os.path.join(fold_path, 'test_index.npy'))
        test_dataset = DiseaseDataset(data_path, test_index)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model_path = os.path.join(save_dir, '6','log', 'checkpoint.pth')
        model = MainModule(input_size, num_heads, ff_dim, num_classes, dis_emb)
        model.to(device)
        loss_fn = torch.nn.BCELoss()
        result_txt = os.path.join(save_dir, 'test_result.txt')
        print('Begin Testing'+'-' * 70)
        test(model, test_loader, loss_fn, indices_tensor, model_path, device, result_txt)
        print('End Testing' + '-' * 70)


