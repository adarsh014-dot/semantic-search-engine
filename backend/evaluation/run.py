"""CLI:  python -m backend.evaluation.run [--dataset PATH] [--k 5]

Evaluates the index currently on disk (upload + index the sample documents first).
Prints real measurements only; nothing is cached or invented.
"""
import argparse
import json

from backend.config import get_settings
from backend.container import build_services
from backend.evaluation.evaluator import DEFAULT_DATASET, evaluate, load_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--quiet", action="store_true", help="omit per-query details")
    args = ap.parse_args()
    svc = build_services(get_settings())
    report = evaluate(svc.search, load_dataset(args.dataset), k=args.k)
    if args.quiet:
        report.pop("per_query")
    print(json.dumps(report, indent=2))
