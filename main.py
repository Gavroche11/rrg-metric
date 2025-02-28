import pandas as pd
import numpy as np
import rrg_metric
import json
import argparse
from typing import Any, List, Dict
import torch.distributed as dist
from pathlib import Path
from rrg_metric.distributed_utils import check_distributed_init, is_main_process
from rrg_metric import AVAILABLE_METRICS

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
    
def main():
    parser = argparse.ArgumentParser(description='Compute metrics for datasets')
    parser.add_argument('--model_ver', type=str, required=True,
                        help='Model version to use for metrics computation')
    parser.add_argument('--hf_cache_dir', type=str, default='/data1/huggingface',
                        help='Huggingface cache directory (default: /data1/huggingface)')
    parser.add_argument('--datasets', nargs='*', default=['snu', 'mimic', 'openi', 'rexgradient'],
                        help='List of datasets to process (default: snu mimic openi rexgradient)')
    parser.add_argument('--test_outputs_root', type=str, default='/data1/workspace/bih1122/llava_test_outputs',
                        help='Root directory for test outputs (default: /data1/workspace/bih1122/llava_test_outputs)')
    parser.add_argument('--metrics_root', type=str, default='/data1/workspace/bih1122/test_metrics',
                        help='Root directory for metrics (default: /data1/workspace/bih1122/test_metrics)')
    args = parser.parse_args()

    test_outputs_dir: Path = Path(args.test_outputs_root) / args.model_ver
    metrics_dir: Path = Path(args.metrics_root) / args.model_ver

    check_distributed_init()
    
    if is_main_process():
        metrics_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        basename = f'{dataset}_test_outputs.csv'
        if (test_outputs_dir / basename).exists():
            if is_main_process():
                print(f"Computing metrics for {dataset}...")
            
            df = pd.read_csv(test_outputs_dir / basename)
            preds: List[str] = df['pred'].tolist()
            gts: List[str] = df['gt'].tolist()

            total_results_dict = {}
            per_sample_results_dict = {}
            green_df = pd.DataFrame()
            for metric in AVAILABLE_METRICS:
                cur_result_dict = {}
                results = rrg_metric.compute(metric=metric,
                                             preds=preds,
                                             gts=gts,
                                             per_sample=False,
                                             verbose=(metric == "green"),
                                             cache_dir=args.hf_cache_dir)

                if is_main_process():
                    if metric != "chexbert":
                        total_results_dict[metric] = results["total_results"]
                        print(f"Computed {metric} results: {results['total_results']}")
                        per_sample_results_dict[metric] = results["per_sample_results"]
                        if metric == "green":
                            green_df = results.get("green_result_df", pd.DataFrame())
                    else:
                        total_results_dict["f1chexbert"] = {
                            "macro_f1_14": results["f1chexbert_macro_f1_14"],
                            "micro_f1_14": results["f1chexbert_micro_f1_14"],
                            "macro_f1_5": results["f1chexbert_macro_f1_5"],
                            "micro_f1_5": results["f1chexbert_micro_f1_5"]
                        }

                        total_results_dict["sembscore"] = results["total_results"]["sembscore"]
                        print(f"Computed f1chexbert results:")
                        print(f"  F1 CheXbert Macro F1 14: {results['f1chexbert_macro_f1_14']}")
                        print(f"  F1 CheXbert Macro F1 5: {results['f1chexbert_macro_f1_5']}")
                        print(f"  F1 CheXbert Micro F1 14: {results['f1chexbert_micro_f1_14']}")
                        print(f"  F1 CheXbert Micro F1 5: {results['f1chexbert_micro_f1_5']}")
                        print(f"Computed sembscore: {results['total_results']['sembscore']}")
                        per_sample_results_dict["f1chexbert"] = results["per_sample_results"]

            if is_main_process():
                total_save_path = metrics_dir / f"{dataset}_test_outputs_metrics.json"
                per_sample_save_path = metrics_dir / f"{dataset}_test_outputs_per_sample_metrics.json"
                json.dump(make_json_serializable(total_results_dict), open(total_save_path, "w"), indent=2)
                print(f"Results for {dataset} saved in {total_save_path}")
                json.dump(make_json_serializable(per_sample_results_dict), open(per_sample_save_path, "w"), indent=2)
                print(f"Per-sample results for {dataset} saved in {per_sample_save_path}")
                if not green_df.empty:
                    green_df_save_path = metrics_dir / f"{dataset}_test_outputs_green_detailed.csv"
                    green_df.to_csv(green_df_save_path, index=False)
                    print(f"Green detailed results for {dataset} saved in {green_df_save_path}")

if __name__ == "__main__":
    main()
