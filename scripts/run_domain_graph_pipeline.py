#!/usr/bin/env python3
"""Thin wrapper → python -m rwcite.cli.graph_pipeline"""
import runpy
import sys
if __name__ == "__main__":
    sys.argv[0] = "rwcite.cli.graph_pipeline"
    runpy.run_module("rwcite.cli.graph_pipeline", run_name="__main__")
