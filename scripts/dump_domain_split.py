#!/usr/bin/env python3
"""Thin wrapper → python -m rwcite.cli.dump_split"""
import runpy
import sys
if __name__ == "__main__":
    sys.argv[0] = "rwcite.cli.dump_split"
    runpy.run_module("rwcite.cli.dump_split", run_name="__main__")
