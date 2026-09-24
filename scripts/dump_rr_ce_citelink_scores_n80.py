#!/usr/bin/env python3
"""Thin wrapper → python -m rwcite.cli.dump_ce_citelink_scores"""
import runpy
import sys
if __name__ == "__main__":
    sys.argv[0] = "rwcite.cli.dump_ce_citelink_scores"
    runpy.run_module("rwcite.cli.dump_ce_citelink_scores", run_name="__main__")
