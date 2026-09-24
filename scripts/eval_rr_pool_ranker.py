#!/usr/bin/env python3
"""Thin wrapper → python -m rwcite.cli.eval_pool_ranker"""
import runpy
import sys
if __name__ == "__main__":
    sys.argv[0] = "rwcite.cli.eval_pool_ranker"
    runpy.run_module("rwcite.cli.eval_pool_ranker", run_name="__main__")
