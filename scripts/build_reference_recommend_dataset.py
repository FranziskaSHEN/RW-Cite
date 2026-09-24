#!/usr/bin/env python3
"""Thin wrapper → python -m rwcite.cli.build_rr_jsonl"""
import runpy
import sys
if __name__ == "__main__":
    sys.argv[0] = "rwcite.cli.build_rr_jsonl"
    runpy.run_module("rwcite.cli.build_rr_jsonl", run_name="__main__")
