import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from config import get_args
from model import MainModule
from data_loader import load_data
from time import time
from tqdm import tqdm



def train(model, num_epochs, train_dataloader, val_dataloader, optimizer, multi_label_loss, indices_tensor, device, save_paths):         
  
    best_valid_loss = float('inf')
    best_model_state = None
    
    t0 = time()
    
    print('Begin Training'+'-' * 70)

    for epoch in range(num_epochs):

        model.train()
        total_loss = 0.0
        train_truth, train_preds, train_labels = [], [], []
        
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

        train_loss = total_loss/len(train_dataloader)

        #模型验证
        model.eval()
        val_loss = 0.0
        total_auc = 0.0
        val_truth, val_preds, val_labels = [], [], []
        
        metrics_dict = {'recall':[],'precision':[],'accuracy':[],'auc':[],'sp':[],'mcc':[], 
               'ap':[]}
        
        # metrics_dict = {'acc': 0, 'auc': 0, 'ap': 0}  
        
        with torch.no_grad():
            for ref_emb, mut_emb, y_class,llm in val_dataloader:
                ref_emb, mut_emb, y_class, llm = ref_emb.to(device), mut_emb.to(device),  y_class.to(device), llm.to(device)
                current_batch_size = ref_emb.size(0)
                indices_tensor = indices_tensor.to(device)
                indices_tensor_disease = indices_tensor.repeat(current_batch_size, 1)

                output_emb = model(ref_emb, mut_emb, indices_tensor_disease,llm)
                val_output = model.classifier(output_emb)

                val_label = (val_output.cpu().detach().numpy() > 0.5).astype(int)

                loss = classification_loss
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        
        print("===>Epoch %02d: loss=%.5f, val_loss=%.4f, time=%ds"
              %(epoch+1, train_loss, val_loss, time()-ti))
        

        # 保存最佳模型
        
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_path)
    
        print('Trained model saved to \'%s/trained_model.h5\'' % (args.save_path))
        print("Total time = %ds" % (time() - t0))
        print('End Training' + '-' * 70)
        
    return best_model_state



def test(model, test_dataloader, multi_label_loss, indices_tensor, save_path, device, output_file):
    
    print('Begin Testing'+'-' * 70)
    
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    # model.cuda()
    model.eval()
    test_loss = 0        
    test_truth, test_preds, test_labels = [], [], []
    all_output_embs = []
    
    with torch.no_grad():
        for ref_emb, mut_emb, y_class,llm in test_dataloader:
            ref_emb, mut_emb, y_class,llm = ref_emb.to(device), mut_emb.to(device), y_class.to(device), llm.to(device)
            
            current_batch_size = ref_emb.size(0)
            indices_tensor = indices_tensor.to(device)
            indices_tensor_disease = indices_tensor.repeat(current_batch_size, 1)
            
            output_emb = model(ref_emb, mut_emb, indices_tensor_disease,llm)
            output = model.classifier(output_emb)

     
            label = (output.cpu().detach().numpy() > 0.5).astype(int)
            
            classification_loss = multi_label_loss(output, y_class)
            loss = classification_loss
            test_loss += loss.item()
            
            test_truth.append(y_class.cpu().numpy())
            test_preds.append(output.cpu().numpy())
            test_labels.append(label)

    
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
        
    print('Test metric saved to \'%s/test.txt\'' % (args.output_file))
    print('End Testing' + '-' * 70)
    

def main():
    
    args = get_args()
    
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
                                               args.save_path )

    # Test the model
    elif args.mode =='test':
        test(model, test_dataloader, loss, indices_tensor, args.save_path, args.device, output_file='test_log.txt')
        

if __name__ == "__main__":
    main()
