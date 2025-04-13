import os
import re

log_dir = "log/covid"
prototypes = [4, 8, 16]

results = {n: {"roc_auc": [], "f1": []} for n in prototypes}

for folder in os.listdir(log_dir):
    if "ProtoCell" in folder:
        n_proto = int(folder.split("_")[1])
        log_path = os.path.join(log_dir, folder, "log.txt")
        with open(log_path, "r") as file:
            content = file.read()
            matches = re.findall(r"\[Evaluation on Test Set\].*?ROC AUC Score: ([0-9.]+) \| F1 Score: ([0-9.]+)", content, re.DOTALL)
            if matches:
                # roc_auc, f1 = matches[-1]   # 마지막 테스트 결과
                roc_auc, f1 = matches[0]    # covid 데이터는, 첫번째 테스트 결과 사용
                results[n_proto]["roc_auc"].append(float(roc_auc))
                results[n_proto]["f1"].append(float(f1))

for n in prototypes:
    roc_auc_avg = sum(results[n]["roc_auc"]) / len(results[n]["roc_auc"])
    f1_avg = sum(results[n]["f1"]) / len(results[n]["f1"])
    print(f"Prototype {n} - ROC AUC Avg: {roc_auc_avg:.2f}, F1 Avg: {f1_avg:.2f}")

"""
Prototype 4 - ROC AUC Avg: 0.88, F1 Avg: 0.63
Prototype 8 - ROC AUC Avg: 0.94, F1 Avg: 0.74
Prototype 16 - ROC AUC Avg: 0.88, F1 Avg: 0.66
"""