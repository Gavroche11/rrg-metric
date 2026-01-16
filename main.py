import pandas as pd
import numpy as np
import rrg_metric
import json
import os
import argparse
from typing import Any
import torch.distributed as dist

def make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0
    
def main():
    METRICS = rrg_metric.AVAILABLE_METRICS

    parser = argparse.ArgumentParser(description='Compute metrics for datasets')
    parser.add_argument('--model_ver', type=str, required=True,
                        help='Model version to use for metrics computation')
    parser.add_argument('--datasets', nargs='*', default=['snu', 'mimic', 'openi', 'rexgradient'],
                        help='List of datasets to process (default: snu mimic openi rexgradient)')
    parser.add_argument('--test_outputs_root', type=str, default='/data1/workspace/bih1122/llava_test_outputs',
                        help='Root directory for test outputs (default: /data1/workspace/bih1122/llava_test_outputs)')
    parser.add_argument('--metrics_root', type=str, default='/data1/workspace/bih1122/test_metrics',
                        help='Root directory for metrics (default: /data1/workspace/bih1122/test_metrics)')
    args = parser.parse_args()

    test_outputs_dir = os.path.join(args.test_outputs_root, args.model_ver)
    metrics_dir = os.path.join(args.metrics_root, args.model_ver)
    
    if is_main_process():
        os.makedirs(metrics_dir, exist_ok=True)

    for dataset in args.datasets:
        basename = f'{dataset}_test_outputs.csv'
        if os.path.exists(os.path.join(test_outputs_dir, basename)):
            if is_main_process():
                print(f"Computing metrics for {dataset}...")
            
            df = pd.read_csv(os.path.join(test_outputs_dir, basename))
            preds = df['pred'].tolist()
            gts = df['gt'].tolist()

            results_dict = {}
            for metric in METRICS:
                cur_result_dict = {}
                results = rrg_metric.compute(metric=metric, preds=preds, gts=gts, per_sample=False, verbose=(metric == "green"), cache_dir="/data1/huggingface")

                if is_main_process():
                    if metric != "chexbert":
                        results_dict[metric] = results["total_results"]
                        print(f"Computed {metric} results: {results['total_results']}")
                    else:
                        results_dict["f1chexbert"] = {
                            "macro_f1_14": results["f1chexbert_macro_f1_14"],
                            "micro_f1_14": results["f1chexbert_micro_f1_14"],
                            "macro_f1_5": results["f1chexbert_macro_f1_5"],
                            "micro_f1_5": results["f1chexbert_micro_f1_5"]
                        }

                        results_dict["sembscore"] = results["total_results"]["sembscore"]
                        print(f"Computed f1chexbert results:")
                        print(f"  F1 CheXbert Macro F1 14: {results['f1chexbert_macro_f1_14']}")
                        print(f"  F1 CheXbert Macro F1 5: {results['f1chexbert_macro_f1_5']}")
                        print(f"  F1 CheXbert Micro F1 14: {results['f1chexbert_micro_f1_14']}")
                        print(f"  F1 CheXbert Micro F1 5: {results['f1chexbert_micro_f1_5']}")
                        print(f"Computed sembscore: {results['total_results']['sembscore']}")

            if is_main_process():
                save_path = os.path.join(metrics_dir, f"{dataset}_test_outputs_metrics.json")
                json.dump(make_json_serializable(results_dict), open(save_path, "w"), indent=4)
                print(f"Results for {dataset} saved in {save_path}")

if __name__ == "__main__":
    main()