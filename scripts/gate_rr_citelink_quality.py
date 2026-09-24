#!/usr/bin/env python3
"""Thin wrapper → python -m rwcite.cli.gate_citelink"""
import runpy
import sys
if __name__ == "__main__":
    sys.argv[0] = "rwcite.cli.gate_citelink"
    runpy.run_module("rwcite.cli.gate_citelink", run_name="__main__")
