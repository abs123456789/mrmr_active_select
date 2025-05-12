import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import f_classif

# ===================== 参数设置 =====================
N  = 20   # 样本数
M0 = 10   # player 0 特征数
M1 = 9    # player 1 特征数
M2 = 10   # player 2 特征数
M3 = 9    # player 3 特征数
K  = 5    # 每个被动方 Top-K
epsilon = 1e-6  # 防止除以0

Ms = [M0, M1, M2, M3]
total_M = sum(Ms)

# 每个 party 的特征全局范围
party_ranges = []
start_idx = 0
for m in Ms:
    party_ranges.append((start_idx, start_idx + m))
    start_idx += m
# party_ranges == [(0,10), (10,19), (19,29), (29,38)]

# ===================== 数据加载 =====================
#data = np.loadtxt("combined.csv", delimiter=",")
data = np.loadtxt("merged_20_39.csv", delimiter=",")

# 每行一个样本：前 total_M 列是特征，最后一列是标签
features = data[:, :-1]           # shape = (20, total_M)
labels   = data[:, -1].astype(int)# shape = (20, )

assert features.shape == (N, total_M)
assert labels.shape   == (N,)

# ===================== 计算每个特征的相关性分数（relevance） =====================
F_values, _ = f_classif(features, labels)
relevance_scores = F_values  # length = total_M

# —— 输出每个特征的 F-score —— #
print("=== 每个特征的 ANOVA F-score ===")
for idx, score in enumerate(relevance_scores):
    print(f"Feature {idx}: F-score = {score:.4f}")

# # ===================== 冗余矩阵计算：相关性绝对值 =====================
# # 计算 Pearson 相关性矩阵（静态方式，与 MPC 对齐）
# corr_matrix = np.corrcoef(features, rowvar=False)  # shape = (total_M, total_M)
# redundancy = np.abs(corr_matrix)
# np.fill_diagonal(redundancy, 0.0)  # 去除自身冗余
# redundancy = np.nan_to_num(redundancy, nan=0.0, posinf=0.0, neginf=0.0)
#
# # ===================== 贪心 mRMR 特征选择 =====================
# selected_by_party = {1: [], 2: [], 3: []}
#
# for pid in [1, 2, 3]:
#     selected_local = list(range(*party_ranges[0]))  # 初始为主动方特征索引
#     start, end = party_ranges[pid]
#
#     for k in range(K):
#         best_score = -np.inf
#         best_feat = None
#
#         idxs = np.array(selected_local, dtype=int)  # 当前已选集合
#
#         for f in range(start, end):
#             if f in selected_local:
#                 continue
#
#             rel = relevance_scores[f]
#             if k == 0:
#                 score = rel
#             else:
#                 denom = np.mean(redundancy[f, idxs])
#                 score = rel / (denom + epsilon)
#
#             if score > best_score:
#                 best_score = score
#                 best_feat = f
#
#         selected_by_party[pid].append(best_feat)
#         selected_local.append(best_feat)

# ===================== 贪心 mRMR 选择：局部冗余性计算 =====================
selected_by_party = {1: [], 2: [], 3: []}

for pid in [1, 2, 3]:
    selected_local = list(range(*party_ranges[0]))  # 主动方特征索引（0~9）
    start, end = party_ranges[pid]                  # 当前被动方特征范围

    for k in range(K):
        best_score = -np.inf
        best_feat  = None

        for f in range(start, end):
            if f in selected_local:
                continue

            x = features[:, int(f)]
            if np.std(x) < 1e-8:
                continue  # 当前候选是常量，跳过

            rel = relevance_scores[f]
            if k == 0:
                score = rel
            else:
                correlations = []
                for j in selected_local:
                    y = features[:, int(j)]
                    if np.std(y) < 1e-8:
                        continue
                    r, _ = pearsonr(x, y)
                    correlations.append(abs(r))
                denom = np.mean(correlations) if correlations else 0.0
                score = rel / (denom + epsilon)

            if score > best_score:
                best_score = score
                best_feat = f

        # round 结束后检查
        if best_feat is None:
            raise ValueError(f"No valid feature selected in round {k} for player {pid}.")

        selected_by_party[pid].append(best_feat)
        selected_local.append(best_feat)

# —— 输出每个被动方的 Top-K 特征索引 —— #
print("\n=== 各被动方 Top-K 特征索引 ===")
for pid in [1, 2, 3]:
    print(f"player{pid}: {selected_by_party[pid]}")