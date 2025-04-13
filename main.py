import pandas as pd
import numpy as np
import rrg_metric
import json
import os

DATASETS = ['mimic', 'openi']
TEST_OUTPUTS_ROOT = '/data1/workspace/bih1122/llava_test_outputs'
METRICS_ROOT = '/data1/workspace/bih1122/test_metrics'

# model_ver = 'v3.2-EVA-X-RRIEF-LLAMA-3.2-3B-3422'
model_ver = "v2.9-full"
test_outputs_dir = os.path.join(TEST_OUTPUTS_ROOT, model_ver)
metrics_dir = os.path.join(METRICS_ROOT, model_ver)
os.makedirs(metrics_dir, exist_ok=True)

available_metrics = rrg_metric.AVAILABLE_METRICS

print("model:", model_ver)
print("test_outputs_dir:", test_outputs_dir)
print("Available metrics:", available_metrics)

for dataset in DATASETS:
    basename = f'{dataset}_test_outputs.csv'
    if os.path.exists(os.path.join(test_outputs_dir, basename)):
        print(f"Computing metrics for {dataset}...")
        df = pd.read_csv(os.path.join(test_outputs_dir, basename))
        preds = df['pred'].tolist()
        gts = df['gt'].tolist()

        results_dict = {}
        for metric in available_metrics:
            cur_result_dict = {}
            results = rrg_metric.compute(metric=metric, preds=preds, gts=gts, per_sample=False, verbose=True)
            if metric in ["bleu", "rouge", "meteor", "bertscore", "f1radgraph"]:
                results_dict[metric] = float(results['total_results'])
            elif metric == "f1chexbert":
                results_dict[metric] = {
                    "accuracy": results['accuracy'],
                    # "accuracy_not_averaged": results['accuracy_not_averaged'].tolist(),
                    "macro_f1_14": results['macro_f1_14'],
                    "micro_f1_14": results['micro_f1_14'],
                    "macro_f1_5": results['macro_f1_5'],
                    "micro_f1_5": results['micro_f1_5'],
                }
            
            print(metric, results_dict[metric])

        save_path = os.path.join(metrics_dir, f"{dataset}_test_outputs_metrics.json")

        json.dump(results_dict, open(save_path, "w"))
        print(f"Results for {dataset} saved in {save_path}")