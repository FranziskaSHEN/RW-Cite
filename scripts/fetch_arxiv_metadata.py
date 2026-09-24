#!/usr/bin/env python3
"""Thin wrapper → python -m rwcite.cli.fetch_metadata"""
import runpy
import sys

if __name__ == "__main__":
    sys.argv[0] = "rwcite.cli.fetch_metadata"
    runpy.run_module("rwcite.cli.fetch_metadata", run_name="__main__")
