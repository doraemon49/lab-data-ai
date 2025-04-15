import os
import re
import statistics

log_dir = "log/icb_ours"
prototypes = [4, 8, 16]
results = {n: {"roc_auc": [], "f1": [], "accuracy": []} for n in prototypes}

for folder in os.listdir(log_dir):
    if "ProtoCell" in folder:
        n_proto = int(folder.split("_")[1])
        log_path = os.path.join(log_dir, folder, "log.txt")
        with open(log_path, "r") as file:
            content = file.read()
            matches = re.findall(
                r"\[Evaluation on Test Set\].*?Avg\. Test Accuracy: ([0-9.]+) \| Avg\. ROC AUC Score: ([0-9.]+) \| F1 Score: ([0-9.]+)",
                content,
                re.DOTALL
            )
            if matches:
                accuracy, roc_auc, f1 = matches[-1]
                results[n_proto]["accuracy"].append(float(accuracy))
                results[n_proto]["roc_auc"].append(float(roc_auc))
                results[n_proto]["f1"].append(float(f1))

for n in prototypes:
    if results[n]["accuracy"]:
        for metric in ["accuracy", "roc_auc", "f1"]:
            values = results[n][metric]
            avg = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"Prototype {n} - {metric.capitalize()} = {avg:.2f} ± {std:.2f}")
    else:
        print(f"Prototype {n} - No data found!")