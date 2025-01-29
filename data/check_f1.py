import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

# 文件路径
submit_path = './SiChuan/submit.csv'
test_true_path = './SiChuan/test_true.csv'
output_path = './SiChuan/result/xgboost_results.txt'

# 读取 CSV 文件
submit_df = pd.read_csv(submit_path)
test_true_df = pd.read_csv(test_true_path)

# 合并数据，确保 phone_no_m 对齐
merged_df = pd.merge(submit_df, test_true_df, on='phone_no_m', suffixes=('_pred', '_true'))

# 计算各项指标
precision = precision_score(merged_df['is_sa_true'], merged_df['is_sa_pred'])
recall = recall_score(merged_df['is_sa_true'], merged_df['is_sa_pred'])
f1 = f1_score(merged_df['is_sa_true'], merged_df['is_sa_pred'])
accuracy = accuracy_score(merged_df['is_sa_true'], merged_df['is_sa_pred'])
auc = roc_auc_score(merged_df['is_sa_true'], merged_df['is_sa_pred'])

# 打印结果
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")

# 保存结果到文件
with open(output_path, 'w') as file:
    file.write("Evaluation Metrics:\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1 Score: {f1:.4f}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"AUC: {auc:.4f}\n")

print(f"Metrics saved to {output_path}")
