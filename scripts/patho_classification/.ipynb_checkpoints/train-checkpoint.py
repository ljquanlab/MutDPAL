import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, recall_score, precision_score, f1_score, confusion_matrix
from tqdm import tqdm
import copy

def train(model, num_epochs, train_dataloader, val_dataloader, optimizer, loss_fn, indices_tensor, device, save_path):
   
    best_valid_loss = float('inf')
    best_model_state = None
    best_result = 0

    print('Begin Training'+'-' * 70)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for ref_emb, mut_emb, y_class, llm, other in tqdm(train_dataloader):
            optimizer.zero_grad()
            ref_emb, mut_emb, y_class, llm, other = ref_emb.to(device), mut_emb.to(device), y_class.to(device), llm.to(device), other.to(device)
            indices_tensor = indices_tensor.to(device)
            indices_tensor_disease = indices_tensor.repeat(ref_emb.size(0), 1)
            output_emb = model(ref_emb, mut_emb, indices_tensor_disease, llm, other)
            output = model.classifier(output_emb)
            loss = loss_fn(output, y_class)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_truth, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for ref_emb, mut_emb, y_class, llm, other in val_dataloader:
                ref_emb, mut_emb, y_class, llm, other = ref_emb.to(device), mut_emb.to(device), y_class.to(device), llm.to(device), other.to(device)
                indices_tensor_disease = indices_tensor.repeat(ref_emb.size(0), 1)
                output_emb = model(ref_emb, mut_emb, indices_tensor_disease, llm, other)
                output = model.classifier(output_emb)
                label = (output.cpu().detach().numpy() > 0.5).astype(int)
                loss = loss_fn(output, y_class)
                val_loss += loss.item()
                
                val_truth.append(y_class.detach().cpu().numpy())
                val_preds.append(output.detach().cpu().numpy())
                val_labels.append(label)
                
        val_loss /= len(val_dataloader)
        
        y_classes = np.concatenate(val_truth, axis=0)
        y_scores = np.concatenate(val_preds, axis=0)
        y_labels = np.concatenate(val_labels, axis=0)
        
        auc = roc_auc_score(y_classes, y_scores)
        mcc = matthews_corrcoef(y_classes, y_labels)
        
        print("===>Epoch %02d: loss=%.5f, val_loss=%.4f"
              %(epoch+1, train_loss, val_loss))
        
        # Save best model
        if 0.5*(auc+mcc) > best_result:
            best_result = 0.5*(auc+mcc)
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_path)
            
        print('Trained model saved to \'%s/trained_model.h5\'' % (save_path))
    print('End Training' + '-' * 70)
            
            
def test(model, test_dataloader, loss_fn, indices_tensor, save_path, device, output_file):
    
    print('Begin Testing'+'-' * 70)
    
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    model.eval()
    test_loss = 0.0
    test_truth, test_preds, test_labels = [], [], []
    with torch.no_grad():
        for ref_emb, mut_emb, y_class, llm, other in test_dataloader:
            ref_emb, mut_emb, y_class, llm, other = ref_emb.to(device), mut_emb.to(device), y_class.to(device), llm.to(device), other.to(device)
            indices_tensor = indices_tensor.to(device)
            indices_tensor_disease = indices_tensor.repeat(ref_emb.size(0), 1)
            output_emb = model(ref_emb, mut_emb, indices_tensor_disease, llm, other)
            output = model.classifier(output_emb)
            label = (output.cpu().detach().numpy() > 0.5).astype(int)
            
            loss = loss_fn(output, y_class)
            test_loss += loss.item()
            test_truth.append(y_class.cpu().numpy())
            test_preds.append(output.cpu().numpy())
            test_labels.append(label)
            
    test_loss /= len(test_dataloader)
    
    y_true = np.concatenate(test_truth, axis=0)
    y_pred = np.concatenate(test_preds, axis=0)
    y_label = np.concatenate(test_labels, axis=0)
    
    acc = np.mean(y_label == y_true)
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_label)
    recall = recall_score(y_true, y_label)
    precision = precision_score(y_true, y_label)
    f1 = f1_score(y_true, y_label)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_label).ravel()
    pn = tn+fp
    specificity = tn / pn
    
    with open(output_file, "a") as f: 
        f.write(f'test Loss : {loss:.6f}, ACC: {acc:.6f}, AUC  : {auc:.6f},MCC: {mcc:.6f},Recall  : {recall:6f},Precision  : {precision:.6f},Sp: {specificity:.6f}, F1:{f1:.6f} \n')
    print('End Testing' + '-' * 70)
