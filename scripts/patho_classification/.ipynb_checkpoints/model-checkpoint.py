import torch
import torch.nn as nn


"""  
Protein representation learning module

"""
class ProtFeatModule(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim):
        
        super(ProtFeatModule, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(512*2, 512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        sattn1 = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(sattn1, 4)

    def forward(self, x):
        
        lstm_out, _ = self.lstm1(x)
        x = self.relu1(self.linear1(lstm_out))
        x = self.relu2(self.linear2(x))
        x = self.encoder(x)
        
        return x


"""  

Disease Embedding

"""   
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

    
 
"""  

Transmembrane environment representation learning module  

"""

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

        x = torch.cat((x1, x2),dim=1)
        return x
    
    
"""
Pathogenic classification module

"""    

class MultiClassification(nn.Module):
    def __init__(self):
        super(MultiClassification,self).__init__()
        self.fc2 = nn.Linear(512+256+256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x = torch.mean(x, dim=1, keepdim=True)
#         x = self.relu1(self.fc1(x))
#         x = x.transpose(1, 2)
        x= self.relu2(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        # x = x.squeeze(2)
        return x
    
    

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
        output = torch.cat((dis_prot, diff1), dim=2)
        output = output.view(output.size(0), 2*256)
        llm_emb = self.llm(llm,llm)
        concat_emb = torch.cat((output, llm_emb),dim=1)
        # concat_emb = torch.cat((output, llm),dim=1)
        # out = self.classifier(concat_emb)
        
        return concat_emb
    