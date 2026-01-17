import pandas as pd
import numpy as np
import rrg_metric
import json
import argparse
from typing import List
from pathlib import Path
from rrg_metric.distributed_utils import check_distributed_init, is_main_process, get_rank
from rrg_metric.utils import make_json_serializable
from rrg_metric import AVAILABLE_METRICS
from loguru import logger

HF_CACHE_DIR = "/data1/huggingface"
DATASETS = ["snu", "mimic", "openi", "rexgradient"]
TEST_OUTPUTS_ROOT = "/data1/workspace/bih1122/llava_test_outputs"
METRICS_ROOT = "/data1/workspace/bih1122/test_metrics"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute metrics for datasets')
    parser.add_argument('--model_ver', type=str, required=True,
                        help='Model version to use for metrics computation')
    parser.add_argument('--hf_cache_dir', type=str, default=HF_CACHE_DIR,
                        help=f'Huggingface cache directory (default: {HF_CACHE_DIR})')
    parser.add_argument('--datasets', nargs='*', default=DATASETS,
                        help=f'List of datasets to process (default: {" ".join(DATASETS)})')
    parser.add_argument('--test_outputs_root', type=str, default=TEST_OUTPUTS_ROOT,
                        help=f'Root directory for test outputs (default: {TEST_OUTPUTS_ROOT})')
    parser.add_argument('--metrics_root', type=str, default=METRICS_ROOT,
                        help=f'Root directory for metrics (default: {METRICS_ROOT})')
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()

    test_outputs_dir: Path = Path(args.test_outputs_root) / args.model_ver
    metrics_dir: Path = Path(args.metrics_root) / args.model_ver

    check_distributed_init()
    rank = get_rank()
    
    if is_main_process():
        metrics_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        basename = f'{dataset}_test_outputs.csv'
        if (test_outputs_dir / basename).exists():
            if is_main_process():
                logger.info(f"[Rank {rank}] Computing metrics for {dataset}...")
            
            df = pd.read_csv(test_outputs_dir / basename)
            preds: List[str] = df['pred'].tolist()
            gts: List[str] = df['gt'].tolist()

            total_results_dict = {}
            per_sample_results_dict = {}
            green_df = pd.DataFrame()
            for metric in AVAILABLE_METRICS:
                results = rrg_metric.compute(metric=metric,
                                             preds=preds,
                                             gts=gts,
                                             per_sample=False,
                                             verbose=(metric == "green"),
                                             cache_dir=args.hf_cache_dir)

                if is_main_process():
                    if metric != "chexbert":
                        total_results_dict[metric] = results["total_results"]
                        logger.info(f"[Rank {rank}] Computed {metric} results: {results['total_results']}")
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
                        logger.info(f"[Rank {rank}] Computed f1chexbert results: "
                                    f"F1 CheXbert Macro F1 14: {results['f1chexbert_macro_f1_14']}, "
                                    f"F1 CheXbert Macro F1 5: {results['f1chexbert_macro_f1_5']}, "
                                    f"F1 CheXbert Micro F1 14: {results['f1chexbert_micro_f1_14']}, "
                                    f"F1 CheXbert Micro F1 5: {results['f1chexbert_micro_f1_5']}")
                        per_sample_results_dict["f1chexbert"] = results["per_sample_results"]

            if is_main_process():
                total_save_path = metrics_dir / f"{dataset}_test_outputs_metrics.json"
                per_sample_save_path = metrics_dir / f"{dataset}_test_outputs_per_sample_metrics.json"
                json.dump(make_json_serializable(total_results_dict), open(total_save_path, "w"), indent=2)
                logger.info(f"[Rank {rank}] Results for {dataset} saved in {total_save_path}")
                json.dump(make_json_serializable(per_sample_results_dict), open(per_sample_save_path, "w"), indent=2)
                logger.info(f"[Rank {rank}] Per-sample results for {dataset} saved in {per_sample_save_path}")
                if not green_df.empty:
                    green_df_save_path = metrics_dir / f"{dataset}_test_outputs_green_detailed.csv"
                    green_df.to_csv(green_df_save_path, index=False)
                    logger.info(f"[Rank {rank}] Green detailed results for {dataset} saved in {green_df_save_path}")

if __name__ == "__main__":
    main()
