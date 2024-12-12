import os
import torch
import pandas as pd
import numpy as np
import pickle
import gc
from tqdm import tqdm
from transformers import EsmTokenizer, EsmModel
import argparse
import warnings
import copy
warnings.filterwarnings('ignore')

# 初始化设备和模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = EsmTokenizer.from_pretrained('../../tools/ESM1v', do_lower_case=False)    #替换为本地ESMn-v存储路径
model = EsmModel.from_pretrained('../../tools/ESM1v').to(device)


# 提取ESM-1v特征 （突变前）
def get_feature(seq):
    seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
    embeddings = []
    
    for i in range(0, len(seq), 1000):
        segment = seq[i:i+1000]
        
        with torch.no_grad():
            token_encoding = tokenizer(segment, return_tensors="pt", padding=True, truncation=True)
            input_ids = token_encoding['input_ids'].to(device)
            attention_mask = token_encoding['attention_mask'].to(device)
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
            segment_emb = embedding_repr.last_hidden_state[0].cpu().numpy()
            embeddings.append(segment_emb)
            
        del input_ids, attention_mask, embedding_repr
        torch.cuda.empty_cache()
        gc.collect()
        
    return np.concatenate(embeddings, axis=0)


# 裁剪ESM-1v特征
def cut_emb(emb, pos, s=50):
    if emb is None:
        return np.zeros((2 * s + 1, 1280))
    
    zero_padding = np.zeros((1, 1280))  # 定义零向量，用于填充
    sequence_length = 2 * s + 1  

    start = max(pos - s, 0)
    end = min(pos + s + 1, len(emb))
    result = emb[start:end]

    padding_needed = sequence_length - result.shape[0]
    if padding_needed > 0:
        padding = np.zeros((padding_needed, 1280))
        if start > 0:
            result = np.vstack((result, padding))
        else:
            result = np.vstack((padding, result))
    return result

def get_emb(df):
    
    embeddings = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        try:
            emb_segment = cut_emb(row.mut_esm_emb, row.pos)
            embeddings.append(emb_segment)
        except KeyError:
            embeddings.append(np.zeros((2 * 50 + 1, 1280)))  # 添加一个与期望形状相同的零向量
            continue

    embeddings = np.stack(embeddings)
    
    return embeddings


# 读取氨基酸的物化性质映射
def load_residue_features(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

    
def cut_seq(seq, pos, mapping, s=50):
    pad = 0
    start = max(pos - s, 0)
    end = min(pos + 1 + s, len(seq))
    result = seq[start : end]
    result = list(map(lambda x: mapping[x], result))
    if pos - s < 0:
        p = s * 2 + 1 - len(result)
        result = [0 for _ in range(p)] + result
        pad = -p
    elif pos + 1 + s > len(seq):
        p = s * 2 + 1 - len(result)
        result += [0 for _ in range(p)]
        pad = p
    return result, pad
    
    
# 提取氨基酸的物理化学属性 
def get_res(df, vocab_mapping, vocab_mapping_r, res_feat):
    r_res = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        try:
            seq, pad = cut_seq(row.mut_seq, row.pos, vocab_mapping)      #替换seq为mut_seq提取突变后的理化性质
        except KeyError:
            print(f"Skipping due to KeyError: seq={row.seq}, pos={row.pos}")
            continue  # Skip this row if there's a KeyError
        
        seq = np.array(seq)
        res = []
        for s in seq:
            if s == 0:
                res.append(np.zeros((213,)))  # If amino acid is 0, append a zero vector
            else:
                res.append(np.array(res_feat[vocab_mapping_r[s]]))  # Use the mapping to get feature
        
        res = np.stack(res)
        r_res.append(res)
    
    return np.stack(r_res)


# 主函数
def main(args):
    # 第一步：读取CSV并提取ESM-1v特征
    print("Reading data...")
    data_df = pd.read_csv(args.input_csv)
    data_df['mut_esm_emb'] = data_df['mut_seq'].apply(get_feature)  #替换seq为突变后序列，提取突变后的语义信息

    # 第二步：裁剪ESM-1v特征
    print("Extracting ESM-1v features...")
    embeddings = get_emb(data_df)
    esm_embeddings = np.array(embeddings, dtype=np.float32)
    np.save(args.esm_output, esm_embeddings)

    # 第三步：提取氨基酸物化性质并裁剪
    print("Processing residue properties...")
    res_feat = load_residue_features(args.res_feat)
            
    vocab = {
    'G', 'P', 'A', 'V',
    'L', 'I', 'M', 'C',
    'F', 'Y', 'W', 'H',
    'K', 'R', 'Q', 'N',
    'E', 'D', 'S', 'T',
    }
    vocab_mapping = {aa: idx for idx, aa in enumerate(sorted(vocab), start=1)}
    vocab_mapping_r = {idx: aa for idx, aa in enumerate(sorted(vocab), start=1)}

    residue_properties = get_res(data_df, vocab_mapping, vocab_mapping_r, res_feat)
    res_float = np.array(residue_properties, dtype=np.float32)
    np.save(args.res_output, res_float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process protein sequences to extract ESM-1v and residue properties features.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--esm_output", type=str, required=True, help="Path to output ESM-1v feature npy file.")
    parser.add_argument("--res_feat", type=str, required=True, help="Path to residue features pickle file.")
    parser.add_argument("--res_output", type=str, required=True, help="Path to output residue properties npy file.")
    args = parser.parse_args()
    main(args)



