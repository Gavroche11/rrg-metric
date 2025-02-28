import pandas as pd
from pathlib import Path
import rrg_metric
import os
import torch
import torch.distributed as dist
from rrg_metric.distributed_utils import is_main_process, check_distributed_init

check_distributed_init()

df = pd.read_csv("test_reports.csv")
preds = df["pred"].tolist()
gts = df["gt"].tolist()

for metric in rrg_metric.AVAILABLE_METRICS:
    results = rrg_metric.compute(
        metric=metric,
        preds=preds,
        gts=gts,
        per_sample=False,
        verbose=(metric == "green"),
        cache_dir="/data1/huggingface"
    )
    if is_main_process():
        print(metric, flush=True)
        print(results["total_results"], flush=True)