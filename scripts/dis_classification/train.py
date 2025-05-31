import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, recall_score, precision_score, f1_score, confusion_matrix
from tqdm import tqdm
import copy


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


def train(model, num_epochs, train_dataloader, val_dataloader, optimizer, loss_fn, indices_tensor, device,num_classes, save_path):
   
    best_valid_loss = float('inf')
    best_model_state = None
    best_result = 0
    threshold_list = [0.5] * num_classes
    print('Begin Training'+'-' * 70)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for ref_emb, mut_emb, y_class, llm in tqdm(train_dataloader):
            optimizer.zero_grad()
            ref_emb, mut_emb, y_class, llm = ref_emb.to(device), mut_emb.to(device), y_class.to(device), llm.to(device)
            indices_tensor = indices_tensor.to(device)
            indices_tensor_disease = indices_tensor.repeat(ref_emb.size(0), 1)
            output_emb = model(ref_emb, mut_emb, indices_tensor_disease, llm)
            output = model.classifier(output_emb)
            loss = loss_fn(output, y_class)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        val_truth, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for ref_emb, mut_emb, y_class, llm in val_dataloader:
                ref_emb, mut_emb, y_class, llm = ref_emb.to(device), mut_emb.to(device), y_class.to(device), llm.to(device)
                indices_tensor_disease = indices_tensor.repeat(ref_emb.size(0), 1)
                output_emb = model(ref_emb, mut_emb, indices_tensor_disease, llm)
                output = model.classifier(output_emb)
                loss = loss_fn(output, y_class)
                val_loss += loss.item()
                val_truth.append(y_class.cpu().numpy())
                val_preds.append(output.cpu().numpy())
                
        val_loss /= len(val_dataloader)
  
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

            f1 = f1_score(y_true, y_label) 
            val_f1 += f1   
            
        val_f1 = val_f1 / num_classes   
        
        # Save best model
        if val_f1 > best_result:
            best_result = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_path)
            
        print("===>Epoch %02d: loss=%.5f, val_loss=%.4f"
              %(epoch+1, train_loss, val_loss))    
        
        print('Trained model saved to \'%s\'' % (save_path))
    print('End Training' + '-' * 70)
    
        
def test(model, test_dataloader, loss_fn, indices_tensor, save_path, device, num_classes, output_file):
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    print('Begin Testing'+'-' * 70)
    
    model.eval()
    test_loss = 0        
    test_truth, test_preds, test_pred_label = [], [], []
    metrics = {'recall':[],'precision':[],'accuracy':[],'auc':[],'sp':[],'mcc':[], 'ap':[]}

    with torch.no_grad():
        for ref_emb, mut_emb, y_class, llm in test_dataloader:
            ref_emb, mut_emb, y_class, llm = ref_emb.to(device), mut_emb.to(device), y_class.to(device), llm.to(device)
            indices_tensor = indices_tensor.to(device)
            indices_tensor_disease = indices_tensor.repeat(ref_emb.size(0), 1)
            
            output_emb = model(ref_emb, mut_emb, indices_tensor_disease, llm)
            output = model.classifier(output_emb)

            loss = loss_fn(output, y_class)
            test_loss += loss.item()
            
            test_truth.append(y_class.cpu().numpy())
            test_preds.append(output.cpu().numpy())
            
    test_loss /= len(test_dataloader)
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
        print('End Testing' + '-' * 70)
        
