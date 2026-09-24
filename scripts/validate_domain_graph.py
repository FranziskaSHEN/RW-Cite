#!/usr/bin/env python3
"""Thin wrapper → python -m rwcite.cli.validate_graph"""
import runpy
import sys
if __name__ == "__main__":
    sys.argv[0] = "rwcite.cli.validate_graph"
    runpy.run_module("rwcite.cli.validate_graph", run_name="__main__")
