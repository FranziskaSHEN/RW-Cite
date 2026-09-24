#!/usr/bin/env python3
"""Thin wrapper → python -m rwcite.cli.rewrite_sentences"""
import runpy
import sys
if __name__ == "__main__":
    sys.argv[0] = "rwcite.cli.rewrite_sentences"
    runpy.run_module("rwcite.cli.rewrite_sentences", run_name="__main__")
