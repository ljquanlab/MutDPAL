import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import get_args
from model import MainModule
from data_loader import load_data
from time import time
from tqdm import tqdm
import numpy as np
from metric_m import precision_multi, recall_multi, f1_multi, hamming_loss
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, hamming_loss, precision_score, recall_score, accuracy_score, coverage_error, auc, roc_auc_score, matthews_corrcoef, roc_curve, precision_recall_curve
import warnings
import copy
warnings.filterwarnings('ignore')



# 训练函数
def train(model, num_epochs, train_dataloader, val_dataloader, optimizer, multi_label_loss, indices_tensor, device, num_classes, save_path):
    
    t0 = time()
    
    print('Begin Training'+'-' * 70)
    
    best_valid_loss = float('inf')
    best_model_state = None
    best_val_f1 = 0.0
    threshold_list = [0.5] * num_classes

    for epoch in range(num_epochs):
        ti = time()
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

        train_loss = total_loss / len(train_dataloader)


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

        for i in range(num_classes):
            y_true = val_y_classes[:, i]
            if not np.any(y_true):
                f.write(f"Skipping class {i+1} due to no positive samples.\n")
                continue  
            y_score = val_y_scores[:, i]
            y_label = (y_score > threshold_list[i]).astype(int)       
            f1 = f1_score(y_true, y_label) 
            val_f1 += f1
                
        val_loss /= len(val_dataloader)
        print("===>Epoch %02d: loss=%.5f, val_loss=%.4f, time=%ds"
              %(epoch+1, train_loss, val_loss, time()-ti))

        # 保存最优模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_path)
            print('Trained model saved to \'%s' % (save_path))
            
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)

    return best_model_state   


def test(model, test_dataloader, multi_label_loss, indices_tensor, save_path, device, num_classes, output_file):
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    
    print('Begin Testing'+'-' * 70)
    
    model.eval()
    test_loss = 0        
    test_truth, test_preds, test_pred_label = [], [],[]
    metrics = {'recall':[],'precision':[],'accuracy':[],'auc':[],'sp':[],'mcc':[], 'ap':[]}

    
    with torch.no_grad():
        for ref_emb, mut_emb, y_class, llm in test_dataloader:
            ref_emb, mut_emb, y_class, llm = ref_emb.to(device), mut_emb.to(device), y_class.to(device), llm.to(device)
            
            current_batch_size = ref_emb.size(0)
            indices_tensor = indices_tensor.to(device)
            indices_tensor_disease = indices_tensor.repeat(current_batch_size, 1)
            
            output_emb = model(ref_emb, mut_emb, indices_tensor_disease, llm)
            output = model.classifier(output_emb)

            classification_loss = multi_label_loss(output, y_class)
            loss = classification_loss
            test_loss += loss.item()
            
            test_truth.append(y_class.cpu().numpy())
            test_preds.append(output.cpu().numpy())

    y_pred_total = np.concatenate(test_preds, axis=0)
    y_true_total = np.concatenate(test_truth, axis=0)
    y_pred_label = np.zeros(y_pred_total.shape)
   
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

        f.write(f"Precision Multi: {precision_multi_:.6f}\n")
        f.write(f"Recall Multi: {recall_multi_:.6f}\n")
        f.write(f"F1 Multi: {f1_multi_:.6f}\n")
        f.write(f"Hamming Loss: {hamming_loss_:.6f}\n")
        
        print('Test metric saved to \'%s' % (output_file))
        print('End Testing' + '-' * 70)
    
    return metrics




def main():
    
    args = get_args()
    print(f"Running mode: {args.mode}")

    
    device = torch.device(args.device)

    # Load dataset
    train_dataloader, val_dataloader, test_dataloader, indices_tensor = load_data(args.data_dir, args.batch_size)

    # Initialize model
    model = MainModule(input_dim=args.input_size, num_heads=args.num_heads, ff_dim=args.ff_dim, 
                       num_classes=args.num_classes, dis_emb=args.dis_emb).to(device)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Define loss function
    loss = nn.BCELoss()

    # Train the model
    if args.mode =='train':
        best_model_state = train(model, args.num_epochs, train_dataloader, val_dataloader,
                                               optimizer, loss, indices_tensor, args.device,
                                               args.num_classes, args.save_path )

    # Test the model
    elif args.mode =='test':
        metrics = test(model, test_dataloader, loss, indices_tensor, args.save_path, args.device, args.num_classes, args.output_path)
        

if __name__ == "__main__":
    main()

