import numpy as np
import torch
from tqdm.auto import tqdm
import os
from typing import List, Dict, Union, Tuple, Any, Optional, Literal
from sklearn.metrics import f1_score
import time
from .distributed_utils import is_distributed, get_rank, get_world_size, split_data, gather_results
import logging
import sys
from loguru import logger

# Suppress PyRuSH debug logging
logging.getLogger("PyRuSH").setLevel(logging.WARNING)
logger.disable("PyRuSH")

def compute(
    metric: Literal["bleu", "rouge", "meteor", "bertscore", "f1radgraph", "chexbert", "ratescore", "green"],
    preds: List[str],
    gts: List[str],
    per_sample: bool = False,
    verbose: bool = False,
    f1radgraph_model_type: Optional[Literal["radgraph", "radgraph-xl", "echograph"]] = "radgraph",
    f1radgraph_reward_level: Optional[Literal["simple", "partial", "complete", "all"]] = "complete",
    cache_dir = None,
) -> Dict[str, Any]:
    """
    Compute evaluation metrics for radiology report generation.
    """
    per_sample_results = None
    additional_results = {}
    assert len(preds) == len(gts), "Number of predictions and ground truths should be the same"

    rank = get_rank()
    logger.info(f"[Rank {rank}] Starting compute for metric: {metric}")

    iters = zip(preds, gts)
    
    def log_msg(msg):
        if verbose:
            logger.info(msg)
        else:
            logger.debug(msg)

    if verbose and metric not in ["f1radgraph", "chexbert", "ratescore", "green"]:
        iters = tqdm(iters, total=len(preds))
        # Optional: update tqdm description via logger or keep as is
    
    if metric in ["bleu", "rouge", "meteor"]:
        from evaluate import load
        
        log_msg(f"Loading '{metric}' computer...")
        computer = load(metric)

        key = "rougeL" if metric == "rouge" else metric

        log_msg(f"Computing '{metric}' scores...")
        start_time = time.time()
        if per_sample:
            per_sample_results = []
            for pred, gt in iters:
                per_sample_result = computer.compute(
                    predictions=[pred],
                    references=[gt],
                )[key]
                per_sample_results.append(per_sample_result)
            total_results = np.mean(per_sample_results)
        else:
            total_results = computer.compute(
                predictions=preds,
                references=gts,
            )[key]
        end_time = time.time()
        total_time = end_time - start_time

    elif metric == "bertscore":
        from evaluate import load
        
        log_msg(f"Loading '{metric}' computer...")
        computer = load(metric)

        log_msg(f"Computing '{metric}' scores...")
        start_time = time.time()
        per_sample_results = computer.compute(
            predictions=preds,
            references=gts,
            lang="en",
        )["f1"]
        total_results = np.mean(per_sample_results)
        end_time = time.time()
        total_time = end_time - start_time

    elif metric == "f1radgraph":
        from .radgraph_gpu import F1RadGraph
        
        log_msg(f"Loading '{metric}' computer...")
        log_msg(f"Model type: {f1radgraph_model_type}, Reward level: {f1radgraph_reward_level}")
        computer = F1RadGraph(model_type=f1radgraph_model_type, reward_level=f1radgraph_reward_level)

        log_msg(f"Computing '{metric}' scores...")
        start_time = time.time()
        res = computer(hyps=preds, refs=gts)
        end_time = time.time()
        total_time = end_time - start_time
        
        if res[0] is None:
            total_results = None
            per_sample_results = None
            additional_results = {}
        else:
            total_results, per_sample_results, pred_graphs, gt_graphs = res
            additional_results = {
                "pred_graphs": pred_graphs,
                "gt_graphs": gt_graphs
            }

    elif metric == "chexbert":
        from .chexbert import CheXbert
        
        log_msg(f"Loading '{metric}' computer...")
        computer = CheXbert(cache_dir=cache_dir)

        log_msg(f"Computing '{metric}' scores...")
        start_time = time.time()
        res = computer(hyps=preds, refs=gts)
        end_time = time.time()
        total_time = end_time - start_time
        
        if res[0] is None:
            total_results = None
            per_sample_results = None
            additional_results = {}
        else:
            accuracy, accuracy_not_averaged, class_report, class_report_5, sembscores = res

            total_results = {
                "f1chexbert" : class_report_5['micro avg']['f1-score'],
                "sembscore"  : np.mean(sembscores)
            }
            per_sample_results = {
                "f1chexbert" : None,
                "sembscore"  : sembscores
            }
            additional_results = {
                "f1chexbert_accuracy"              : accuracy,
                "f1chexbert_accuracy_not_averaged" : accuracy_not_averaged,
                "f1chexbert_micro_precision_14"    : class_report['micro avg']['precision'],
                "f1chexbert_micro_recall_14"       : class_report['micro avg']['recall'],
                "f1chexbert_micro_f1_14"           : class_report['micro avg']['f1-score'],
                "f1chexbert_micro_precision_5"     : class_report_5['micro avg']['precision'],
                "f1chexbert_micro_recall_5"        : class_report_5['micro avg']['recall'],
                "f1chexbert_micro_f1_5"            : class_report_5['micro avg']['f1-score'],
                "f1chexbert_macro_precision_14"    : class_report['macro avg']['precision'],
                "f1chexbert_macro_recall_14"       : class_report['macro avg']['recall'],
                "f1chexbert_macro_f1_14"           : class_report['macro avg']['f1-score'],
                "f1chexbert_macro_precision_5"     : class_report_5['macro avg']['precision'],
                "f1chexbert_macro_recall_5"        : class_report_5['macro avg']['recall'],
                "f1chexbert_macro_f1_5"            : class_report_5['macro avg']['f1-score'],
            }

    elif metric == "ratescore":
        from RaTEScore import RaTEScore
        log_msg(f"Loading '{metric}' computer...")
        computer = RaTEScore()

        log_msg(f"Computing '{metric}' scores...")
        
        if is_distributed():
             rank = get_rank()
             world_size = get_world_size()
             preds_split = split_data(preds, rank, world_size)
             gts_split = split_data(gts, rank, world_size)
        else:
             preds_split = preds
             gts_split = gts

        logger.info(f"[Rank {rank}] Starting RaTEScore local computation")
        start_time = time.time()
        local_results = computer.compute_score(preds_split, gts_split)
        logger.info(f"[Rank {rank}] Finished RaTEScore local computation")
        
        if isinstance(local_results, list):
            clean_results = []
            for res in local_results:
                if hasattr(res, 'item'):
                     clean_results.append(res.item())
                else:
                     clean_results.append(res)
            local_results = clean_results

        if is_distributed():
            logger.info(f"[Rank {rank}] Entering gather_results for {metric}")
            gathered_results = gather_results(local_results)
            logger.info(f"[Rank {rank}] Finished gather_results for {metric}")
            if get_rank() == 0:
                per_sample_results = gathered_results
                total_results = np.mean(per_sample_results)
            else:
                per_sample_results = None
                total_results = None
        else:
            per_sample_results = local_results
            total_results = np.mean(per_sample_results)
            
        end_time = time.time()
        total_time = end_time - start_time

    elif metric == "green":
        from .green_score import GREEN
        log_msg(f"Loading '{metric}' computer...")
        green_model_name = "StanfordAIMI/GREEN-radllama2-7b"
        computer = GREEN(model_name=green_model_name, cache_dir=cache_dir, output_dir='.', compute_summary_stats=False)

        log_msg(f"Computing '{metric}' scores...")
        start_time = time.time()
        logger.info(f"[Rank {rank}] Calling GREEN computer")
        mean, std, green_score_list, summary, result_df = computer(refs=gts, hyps=preds)
        logger.info(f"[Rank {rank}] GREEN computer returned")
        end_time = time.time()
        total_time = end_time - start_time
        total_results = mean
        per_sample_results = green_score_list
        additional_results = {
            "green_std": std,
            "green_summary": summary,
            "green_result_df": result_df,
        }

    else:
        raise ValueError(f"Invalid metric: {metric}")

    logger.info(f"[Rank {rank}] Starting cleanup")
    if 'computer' in locals():
        del computer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        logger.debug(f"[Rank {rank}] Emptying cache")
        torch.cuda.empty_cache()
    logger.info(f"[Rank {rank}] Cleanup finished")

    log_msg("Done.")

    return {
        "total_results": total_results,
        "per_sample_results": per_sample_results,
        "total_time": total_time,
        **additional_results
    }
