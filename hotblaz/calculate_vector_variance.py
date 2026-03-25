import numpy as np
from collections import defaultdict
import torch

def read_token_vectors(file_path):
    """读取 token vector 文件，返回 {token: vector} 字典"""
    token_vectors = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            token, vector_str = line.strip().split("\t")
            vector = np.array([float(v) for v in vector_str.split(", ")])
            token_vectors[token] = vector
    return token_vectors

def calculate_euclidean_distance(vec1, vec2):
    """计算两个向量的欧氏距离"""
    return np.linalg.norm(vec1 - vec2)

def calculate_cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_euclidean_distance(vec1, vec2):
    # Convert lists to numpy arrays if they aren't already
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2)

'''
def Eucli_Dist(vectors1, vectors2, embedding_dim=768):
    all_tokens = set(vectors1.keys()) | set(vectors2.keys())  # 取并集
    if not all_tokens:
        print("Error: No tokens to compare!")
        return None
    
    euclidean_distances = []
    
    for token in all_tokens:
        vec1 = vectors1.get(token, [0.0] * embedding_dim)  # 缺失时用零向量填充
        vec2 = vectors2.get(token, [0.0] * embedding_dim)
        euclidean_dist = calculate_euclidean_distance(vec1, vec2)
        euclidean_distances.append(euclidean_dist)
    
    avg_euclidean = np.mean(euclidean_distances)
    return avg_euclidean

'''

def Eucli_Dist(ids1, emb1, ids2, emb2):
    """
    输入：
    ids1, ids2: [seq_len]
    emb1, emb2: [seq_len, 768]

    逻辑完全复刻旧版：逐 token 计算欧氏距离
    """
    # 1. 获取 token 总集合 (旧逻辑使用并集)
    all_ids = torch.unique(torch.cat([ids1, ids2]))

    distances = []

    for tid in all_ids:
        # 找到 token 对应的 index（可能不存在）
        idx1 = (ids1 == tid).nonzero(as_tuple=False)
        idx2 = (ids2 == tid).nonzero(as_tuple=False)

        # 不存在 → 使用零向量（旧逻辑）
        if idx1.numel() == 0:
            v1 = torch.zeros((1, emb1.shape[1]), device="cuda")
        else:
            v1 = emb1[idx1[0]]

        if idx2.numel() == 0:
            v2 = torch.zeros((1, emb2.shape[1]), device="cuda")
        else:
            v2 = emb2[idx2[0]]

        # 欧氏距离（GPU）
        diff = v1 - v2
        dist = torch.sqrt((diff * diff).sum())
        distances.append(dist)

    # 取平均（旧逻辑）
    distances_tensor = torch.stack(distances)   # [num_tokens]
    return distances_tensor.mean().item()

def main():
    # 读取两个文件
    file1 = "token_vectors.txt"
    file2 = "token_vectors_sample05.txt"
    
    vectors1 = read_token_vectors(file1)
    vectors2 = read_token_vectors(file2)
    
    # 找出共同的 token
    common_tokens = set(vectors1.keys()) & set(vectors2.keys())
    if not common_tokens:
        print("Error: No common tokens found between the two files!")
        return
    
    # 计算方差（欧氏距离和余弦相似度）
    euclidean_distances = []
    cosine_similarities = []
    
    for token in common_tokens:
        vec1 = vectors1[token]
        vec2 = vectors2[token]
        
        euclidean_dist = calculate_euclidean_distance(vec1, vec2)
        cosine_sim = calculate_cosine_similarity(vec1, vec2)
        
        euclidean_distances.append(euclidean_dist)
        cosine_similarities.append(cosine_sim)
        
        # 可选：打印每个 token 的差异
        # print(f"Token: {token}, Euclidean: {euclidean_dist:.4f}, Cosine: {cosine_sim:.4f}")
    
    # 计算平均值
    avg_euclidean = np.mean(euclidean_distances)
    avg_cosine = np.mean(cosine_similarities)
    
    print(f"Number of common tokens: {len(common_tokens)}")
    print(f"Average Euclidean distance: {avg_euclidean:.4f}")
    print(f"Average Cosine similarity: {avg_cosine:.4f}")

if __name__ == "__main__":
    main()