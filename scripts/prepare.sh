#!/bin/bash

uv run src/main.py
uv run src/prune.py
uv run src/evaluate.py