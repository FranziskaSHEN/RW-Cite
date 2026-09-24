#!/usr/bin/env python3
"""Thin wrapper → python -m rwcite.cli.diag_retrieval"""
import runpy
import sys
if __name__ == "__main__":
    sys.argv[0] = "rwcite.cli.diag_retrieval"
    runpy.run_module("rwcite.cli.diag_retrieval", run_name="__main__")
