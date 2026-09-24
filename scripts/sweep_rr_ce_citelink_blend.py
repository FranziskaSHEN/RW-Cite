#!/usr/bin/env python3
"""Thin wrapper → python -m rwcite.cli.sweep_blend"""
import runpy
import sys
if __name__ == "__main__":
    sys.argv[0] = "rwcite.cli.sweep_blend"
    runpy.run_module("rwcite.cli.sweep_blend", run_name="__main__")
