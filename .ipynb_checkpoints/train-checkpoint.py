import copy
from itertools import cycle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, hamming_loss, precision_score, recall_score, accuracy_score, coverage_error, auc, roc_auc_score, matthews_corrcoef, roc_curve,precision_recall_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')
 
    
class DiseaseDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        
        self.index = np.load(os.path.join(self.data_path, 'multi_index.npy'))
        self.ref_emb = np.load(os.path.join(self.data_path, 'remb1.npy'))
        self.mut_emb = np.load(os.path.join(self.data_path,'memb1.npy'))
        self.ref_res = np.load(os.path.join(self.data_path,'rres.npy'))
        self.mut_res = np.load(os.path.join(self.data_path,'mres.npy'))
        self.y_class = np.load(os.path.join(self.data_path,'y_class1.npy'))
        self.llm = np.load(os.path.join(self.data_path,'llm1.npy'))
            
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

    def forward(self, x1, x2):
        x1 = self.linear1(x1)
        x2 = x2.unsqueeze(-1)
        x2 = self.relu1(self.conv1(x2))
        x2 = self.relu2(self.conv2(x2))
        x2 = x2.squeeze(-1)

        x = torch.cat((x1, x2),dim=1)
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

    
class MultiClassification(nn.Module):
    def __init__(self):
        super(MultiClassification,self).__init__()
        # self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(15*256+256+256, 2048)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2048, 1024)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(1024, 256)
        self.relu4 = nn.ReLU()
        self.fc5  = nn.Linear(256,15)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x    

class TempLossFunction(nn.Module):
    def __init__(self, device):
        super(TempLossFunction, self).__init__()
        self.device = device
        self.lamda = torch.zeros(1, 15, device=self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    def set_active_disease(self, disease_index):
        # 设置当前疾病的权重为1，其他为0
        self.lamda.zero_()
        self.lamda[0, disease_index] = 1.0

    def forward(self, predictions, targets):
        BCELoss = self.criterion(predictions, targets)
        loss = self.lamda * BCELoss
        return loss
  

class MainModule(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_classes, dis_emb):
        super(MainModule, self).__init__()
        self.prot = ProtFeatModule(input_dim=input_dim, num_heads=num_heads, ff_dim=ff_dim)
        self.disease = DiseaseEmbedding(num_classes=num_classes, embedding_dim=dis_emb)
        self.emb1 = CrossAttention(d_model=dis_emb, num_heads=num_heads, kdim=128, vdim=128, ff_dim=ff_dim) 
        self.emb2 = CrossAttention(d_model=dis_emb, num_heads=num_heads, kdim=128, vdim=128, ff_dim=ff_dim) 
        self.llm = LLMEmbedding()
        self.classifier = MultiClassification()
        
    def forward(self, ref_prot, mut_prot, indices_tensor, llm):
        ref = self.prot(ref_prot)
        mut = self.prot(mut_prot) 
        dis_emb = self.disease(indices_tensor)
        ref_emb = self.emb1(dis_emb, ref, ref)
        mut_emb = self.emb1(dis_emb, mut, mut)
        diff1 = ref_emb - mut_emb
        dis_prot = self.emb2(ref_emb, mut_emb, mut_emb)
        output = torch.cat((dis_prot, diff1), dim=-1)
        output = output.view(output.size(0), 15*256)
        llm_emb = self.llm(llm,llm)
        concat_emb = torch.cat((output, llm_emb),dim=1)
        # concat_emb = torch.cat((output, llm),dim=1)
        # out = self.classifier(concat_emb)
        return concat_emb

    
# 根据给定索引调整样本权重  
def get_sample_weights(y_weights, indices):
    return y_weights[indices]


def precision_multi(y_true,y_pred):
    """
        Input: y_true, y_pred with shape: [n_samples, n_classes]
        Output: example-based precision

    """
    n_samples = y_true.shape[0]
    result = 0
    for i in range(n_samples):
        if not (y_pred[i] == 0).all():
            true_posi = y_true[i] * y_pred[i]
            n_true_posi = np.sum(true_posi)
            n_pred_posi = np.sum(y_pred[i])
            result += n_true_posi / n_pred_posi
    return result / n_samples

def recall_multi(y_true,y_pred):
    """
        Input: y_true, y_pred with shape: [n_samples, n_classes]
        Output: example-based recall
    """
    n_samples = y_true.shape[0]
    result = 0
    for i in range(n_samples):
        if not (y_true[i] == 0).all():
            true_posi = y_true[i] * y_pred[i]
            n_true_posi = np.sum(true_posi)
            n_ground_true = np.sum(y_true[i])
            result += n_true_posi / n_ground_true
    return result / n_samples

def f1_multi(y_true,y_pred):
    """
        Input: y_true, y_pred with shape: [n_samples, n_classes]
        Output: example-based recall
    """
    n_samples = y_true.shape[0]
    result = 0
    for i in range(n_samples):
        if not ((y_true[i] == 0).all() and (y_pred[i] == 0).all()):
            true_posi = y_true[i] * y_pred[i]
            n_true_posi = np.sum(true_posi)
            n_ground_true = np.sum(y_true[i])
            n_pred_posi = np.sum(y_pred[i])
            f1 = 2*(n_true_posi) / (n_ground_true+n_pred_posi)
            result += f1
    return result / n_samples

def hamming_loss(y_true,y_pred):
    """
        Input: y_true, y_pred with shape: [n_samples, n_classes]
        Output: hamming loss
    """
    n_samples = y_true.shape[0]
    n_classes = y_true.shape[1]
    loss = 0
    for i in range(n_samples):
        xor = np.sum((y_true[i] + y_pred[i]) % 2)
        loss += xor / n_classes
    return loss / n_samples



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, average_precision_score

def test_dl(model, test_dataloader, multi_label_loss, indices_tensor, save_path, device, num_classes, output_file):
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    
    model.eval()
    test_loss = 0        
    test_truth, test_preds, test_pred_label = [], [],[]
    metrics = {'recall':[],'precision':[],'accuracy':[],'auc':[],'sp':[],'mcc':[], 'ap':[]}
    class_names = ['NSD', 'DSD', 'OCD', 'CDM', 'RSD', 'CD', 'RD', 'ISD', 'EMD', 'MD', 'USD', 'SD', 'CN', 'NP', 'UN']
    all_output_embs = []
    with torch.no_grad():
        for ref_emb, mut_emb, y_class, llm in test_dataloader:
            ref_emb, mut_emb, y_class, llm = ref_emb.to(device), mut_emb.to(device), y_class.to(device), llm.to(device)
            
            current_batch_size = ref_emb.size(0)
            indices_tensor = indices_tensor.to(device)
            indices_tensor_disease = indices_tensor.repeat(current_batch_size, 1)
            
            output_emb = model(ref_emb, mut_emb, indices_tensor_disease, llm)
            output = model.classifier(output_emb)
            all_output_embs.append(output_emb.cpu().numpy())

            classification_loss = multi_label_loss(output, y_class)
            loss = classification_loss
            test_loss += loss.item()
            
            test_truth.append(y_class.cpu().numpy())
            test_preds.append(output.cpu().numpy())

    y_pred_total = np.concatenate(test_preds, axis=0)
    y_true_total = np.concatenate(test_truth, axis=0)
    y_pred_label = np.zeros(y_pred_total.shape)
    all_output_embs = np.concatenate(all_output_embs, axis=0)
    np.save('/public/home/ljquan/clx/mut_train/plot/test_embeddingm.npy',all_output_embs)
    np.save('/public/home/ljquan/clx/mut_train/plot/test_true.npy',y_true_total)
    np.save('/public/home/ljquan/clx/mut_train/plot/test_pred.npy',y_pred_total)
   

    with open(output_file, "a") as f:
        f.write("测试集结果:\n")
        for i in range(num_classes):
            y_true = y_true_total[:, i]
            if not np.any(y_true):
                f.write(f"Skipping class {i+1} due to no positive samples.\n")
                continue  
            y_score = y_pred_total[:, i]
            y_label = (y_score > 0.5).astype(int)
            y_pred_label[:,i] = y_label
            
            acc = np.mean(y_label == y_true)
            auc = roc_auc_score(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            mcc = matthews_corrcoef(y_true, y_label)
            recall = recall_score(y_true, y_label)
            precision = precision_score(y_true, y_label)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_label).ravel()
            specificity = tn / (tn + fp)

            metrics['sp'].append(specificity)
            metrics['accuracy'].append(acc)
            metrics['auc'].append(auc)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['mcc'].append(mcc)
            metrics['ap'].append(ap)
            
            f.write(f"Class {i+1} - Accuracy: {acc:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, MCC: {mcc:.4f}, Specificity: {specificity:.4f}, AP: {ap:.4f}\n")
        precision_multi_ = precision_multi(y_true_total, y_pred_label)
        recall_multi_ = recall_multi(y_true_total, y_pred_label)
        f1_multi_ = f1_multi(y_true_total, y_pred_label)
        hamming_loss_ = hamming_loss(y_true_total, y_pred_label)
        np.save('/public/home/ljquan/clx/mut_train/plot/test_label.npy',y_pred_label)
        f.write(f"Precision Multi: {precision_multi_:.6f}\n")
        f.write(f"Recall Multi: {recall_multi_:.6f}\n")
        f.write(f"F1 Multi: {f1_multi_:.6f}\n")
        f.write(f"Hamming Loss: {hamming_loss_:.6f}\n")
    
# 绘制多标签AUC和AUPR
#     from math import sqrt
#     import matplotlib as mpl
#     golden_mean = (sqrt(5)-1.0)/2.0 #used for size=
#     fig_width = 6 # fig width in inches
#     fig_height = fig_width*golden_mean
#     mpl.rcParams['axes.labelsize'] = 10
#     mpl.rcParams['axes.titlesize'] = 10
#     mpl.rcParams['font.size'] = 10
#     mpl.rcParams['legend.fontsize'] = 10
#     mpl.rcParams['xtick.labelsize'] = 8
#     mpl.rcParams['ytick.labelsize'] = 8
#     mpl.rcParams['text.usetex'] = False
#     mpl.rcParams['font.family'] = 'serif'
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_width*2+1,fig_height+0.1))
#     colors =['#f5a79c','#9eaac4','#c8e7e0','#a7dcf0','#facec4','#ea7d7e','#81cdc3','#c1cbda','#bbb2a3','#fab55c','#f7f5b4','#bbb8d4','#999999','#cce5c5','#d4d8d7']

#     # 绘制ROC曲线
#     for i, class_name in enumerate(['NSD', 'DSD', 'OCD', 'CDM', 'RSD', 'CD', 'RD', 'ISD', 'EMD', 'MD', 'USD', 'SD', 'CN', 'NP', 'UN']):
#         y_true = y_true_total[:, i]
#         y_score = y_pred_total[:, i]
#         fpr, tpr, _ = roc_curve(y_true, y_score)
#         axes[0].plot(fpr, tpr, color=colors[i], lw=2, label=f'{class_name} (AUC = {metrics["auc"][i]:.2f})')

#     axes[0].plot([0, 1], [0, 1], 'k--', lw=2)
#     axes[0].set_xlim([0.0, 1.0])
#     axes[0].set_ylim([0.0, 1.0])
#     axes[0].set_xlabel('False Positive Rate')
#     axes[0].set_ylabel('True Positive Rate')
#     axes[0].set_title('ROC Curves')
#     axes[0].tick_params(axis='x', which='both', top=False)
#     axes[0].tick_params(axis='y', which='both', right=False)
#     axes[0].set_aspect('equal', adjustable='box')

#     # 绘制PR曲线
#     for i, class_name in enumerate(['NSD', 'DSD', 'OCD', 'CDM', 'RSD', 'CD', 'RD', 'ISD', 'EMD', 'MD', 'USD', 'SD', 'CN', 'NP', 'UN']):
#         y_true = y_true_total[:, i]
#         y_score = y_pred_total[:, i]
#         precision, recall, _ = precision_recall_curve(y_true, y_score)
#         axes[1].plot(recall, precision, color=colors[i], lw=2, 
#                      label=f'{class_name} (AUC = {metrics["auc"][i]:.2f}, AP = {metrics["ap"][i]:.2f})')

#     axes[1].set_xlim([0.0, 1.0])
#     axes[1].set_ylim([0.0, 1.0])
#     axes[1].set_xlabel('Recall')
#     axes[1].set_ylabel('Precision')
#     axes[1].set_title('Precision-Recall Curves')
#     axes[1].tick_params(axis='x', which='both', top=False)
#     axes[1].tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
#     axes[1].set_aspect('equal', adjustable='box')

#     # 设置图例在右侧
#     axes[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., frameon=False)

#     # 调整布局并保存图像
#     fig.tight_layout()
#     fig.savefig('/public/home/ljquan/clx/mut_train/plot/multi_label_auc_aupr.png')
    return metrics


# 训练函数
def train_dl(model, num_epochs, train_dataloader, val_dataloader, optimizer, multi_label_loss, multi_loss, device, indices_tensor, num_classes, save_path,output_file):
    best_valid_loss = float('inf')
    best_model_state = None
    best_valid_auc = 0.0
    best_val_f1 = 0.0
    threshold_list = [0.5] * num_classes

    with open(output_file, "a") as f:
        for epoch in range(num_epochs):
            f.write(f'第{epoch+1}轮训练结果:\n')
            model.train()
            total_loss = 0.0

            train_truth, train_preds, train_labels = [], [], []   
            train_metrics_dict = {'recall':[],'precision':[],'accuracy':[],'auc':[],'sp':[],'mcc':[], 'ap':[]}
            
            for ref_emb, mut_emb, y_class, llm in tqdm(train_dataloader):    
                optimizer.zero_grad()
                ref_emb, mut_emb, y_class, llm = ref_emb.to(device), mut_emb.to(device), y_class.to(device), llm.to(device)
                current_batch_size = ref_emb.size(0)

                indices_tensor = indices_tensor.to(device)
                indices_tensor_disease = indices_tensor.repeat(current_batch_size, 1)

                output_emb = model(ref_emb, mut_emb, indices_tensor_disease,llm)
                output = model.classifier(output_emb)

                classification_loss = multi_label_loss(output, y_class)
                classification_loss = classification_loss.mean()
                loss = classification_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                train_truth.append(y_class.detach().cpu().numpy())
                train_preds.append(output.detach().cpu().numpy())

            train_y_classes = np.concatenate(train_truth, axis=0)
            train_y_scores = np.concatenate(train_preds, axis=0)
            train_y_labels = np.zeros(train_y_scores.shape)

            train_loss = total_loss / len(train_dataloader)

            for i in range(num_classes):
                y_true = train_y_classes[:, i]
                if not np.any(y_true):
                    f.write(f"Skipping class {i+1} due to no positive samples.\n")
                    continue  
                y_score = train_y_scores[:, i]
                y_label = (y_score > 0.5).astype(int)
                train_y_labels[:,i] = y_label
                acc = np.mean(y_label == y_true)
                auc = roc_auc_score(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
                mcc = matthews_corrcoef(y_true, y_label)
                recall = recall_score(y_true, y_label)
                precision = precision_score(y_true, y_label)

                tn, fp, fn, tp = confusion_matrix(y_true, y_label).ravel()
                specificity = tn / (tn + fp)

                train_metrics_dict['sp'].append(specificity)
                train_metrics_dict['accuracy'].append(acc)
                train_metrics_dict['auc'].append(auc)
                train_metrics_dict['precision'].append(precision)
                train_metrics_dict['recall'].append(recall)
                train_metrics_dict['mcc'].append(mcc)
                train_metrics_dict['ap'].append(ap)

                f.write(f"Class {i+1} - Accuracy: {acc:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, MCC: {mcc:.4f}, Specificity: {specificity:.4f}, AP: {ap:.4f}\n")

            f.write(f"Train Loss: {train_loss:.4f}\n")
            
            precision_multi_ = precision_multi(train_y_classes, train_y_labels)
            recall_multi_ = recall_multi(train_y_classes, train_y_labels)
            f1_multi_ = f1_multi(train_y_classes, train_y_labels)
            hamming_loss_ = hamming_loss(train_y_classes, train_y_labels)
            
            f.write(f"Train Precision Multi: {precision_multi_:.6f}\n")
            f.write(f"Train Recall Multi: {recall_multi_:.6f}\n")
            f.write(f"Train F1 Multi: {f1_multi_:.6f}\n")
            f.write(f"Train Hamming Loss: {hamming_loss_:.6f}\n")
            
            # 验证集评估
            model.eval()
            val_loss = 0
            val_auc = 0.0
            val_f1 = 0.0
            val_truth, val_preds = [], []
            val_metrics_dict = {'recall':[],'precision':[],'accuracy':[],'auc':[],'sp':[],'mcc':[], 'ap':[]}
            
            with torch.no_grad():
                for ref_emb, mut_emb, y_class, llm in val_dataloader:
                    ref_emb, mut_emb, y_class, llm = ref_emb.to(device), mut_emb.to(device), y_class.to(device), llm.to(device)
                    
                    current_batch_size = ref_emb.size(0)
                    indices_tensor = indices_tensor.to(device)
                    indices_tensor_disease = indices_tensor.repeat(current_batch_size, 1)
                    
                    output_emb = model(ref_emb, mut_emb, indices_tensor_disease,llm)
                    output = model.classifier(output_emb)

                    classification_loss = multi_label_loss(output, y_class)
                    loss = classification_loss
    
                    val_loss += loss.item()
                    
                    val_truth.append(y_class.cpu().numpy())
                    val_preds.append(output.cpu().numpy())
                
            val_y_classes = np.concatenate(val_truth, axis=0)
            val_y_scores = np.concatenate(val_preds, axis=0)
            val_y_label = np.zeros(val_y_scores.shape)
            
            for i in range(num_classes):
                y_true = val_y_classes[:, i]
                if not np.any(y_true):
                    f.write(f"Skipping class {i+1} due to no positive samples.\n")
                    continue  
                y_score = val_y_scores[:, i]
                y_label = (y_score > 0.5).astype(int)  
                val_y_label[:,i] = y_label
                acc = np.mean(y_label == y_true)
                auc = roc_auc_score(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
                mcc = matthews_corrcoef(y_true, y_label)
                recall = recall_score(y_true, y_label)
                precision = precision_score(y_true, y_label)
                f1 = f1_score(y_true, y_label) 
                
                tn, fp, fn, tp = confusion_matrix(y_true, y_label).ravel()
                specificity = tn / (tn + fp)
                
                val_metrics_dict['sp'].append(specificity)
                val_metrics_dict['accuracy'].append(acc)
                val_metrics_dict['auc'].append(auc)
                val_metrics_dict['precision'].append(precision)
                val_metrics_dict['recall'].append(recall)
                val_metrics_dict['mcc'].append(mcc)
                val_metrics_dict['ap'].append(ap)
                val_auc += auc
                val_f1 += f1
                
                f.write(f"Class {i+1} - Validation Accuracy: {acc:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, MCC: {mcc:.4f}, Specificity: {specificity:.4f}, AP: {ap:.4f}\n")
            
            val_loss = val_loss / len(val_dataloader)
            val_auc = val_auc /num_classes
            val_f1 = val_f1 / num_classes
            f.write(f"Validation Loss: {val_loss:.4f}\n")
            
            precision_multi_val_ = precision_multi(val_y_classes, val_y_label)
            recall_multi_val_ = recall_multi(val_y_classes, val_y_label)
            f1_multi_val_ = f1_multi(val_y_classes, val_y_label)
            hamming_loss_val_ = hamming_loss(val_y_classes, val_y_label)
            
            f.write(f"Validation Precision Multi: {precision_multi_val_:.6f}\n")
            f.write(f"Validation Recall Multi: {recall_multi_val_:.6f}\n")
            f.write(f"Validation F1 Multi: {f1_multi_val_:.6f}\n")
            f.write(f"Validation Hamming Loss: {hamming_loss_val_:.6f}\n")
            
            # 如果当前模型比之前的最优模型更好，则保存当前模型和阈值
            # if val_auc > best_valid_auc:
            #     best_valid_auc = val_auc
            #     best_model_state = copy.deepcopy(model.state_dict())
            #     torch.save(best_model_state, save_path)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, save_path)

    return best_model_state, best_valid_auc   

if __name__ == "__main__":

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
    
    #设置超参数
    input_size = 1280 + 213
    num_heads = 8
    ff_dim = 256
    dis_emb = 128
    num_classes = 15
    num_epochs = 200
    learning_rate = 0.001
    batch_size = 512
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    save_path = '/public/home/ljquan/clx/mut_train/model_state_mm/train_best9.pth'
    data_dir = '/public/home/ljquan/clx/mut_train/data'
    result_txt = '/public/home/ljquan/clx/mut_train/model_state_mm/result9.txt'
    # 疾病embedding
    class_indices = {key:i for i, key in enumerate(y_class.keys())}
    indices = [class_indices[key] for key in y_class.keys()]
    indices_tensor = torch.LongTensor(indices)


    #加载数据集
    dataset = DiseaseDataset(data_dir)    
    train_indices, temp_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_indices), shuffle=False)

    model = MainModule(input_size, num_heads, ff_dim, num_classes, dis_emb)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    multi_label_loss = torch.nn.BCELoss()
    multi_loss = TempLossFunction(device=device)

    # train_state, train_loss = train_dl(model, num_epochs, train_dataloader, val_dataloader, optimizer,
    #                                 multi_label_loss, multi_loss, device,indices_tensor, num_classes,
    #                                 save_path, result_txt)

    metrics_dl = test_dl(model, test_dataloader, multi_label_loss, indices_tensor ,save_path, device, num_classes, result_txt)
    print('Deep Learning Metrics:\n')
    print(metrics_dl)