#!/bin/bash

set -e 

echo "========================================"
echo "Starting Universal Compressor Benchmarks"
echo "Date: $(date)"
echo "========================================"

mkdir -p results

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="results/benchmark_log_${TIMESTAMP}.txt"

echo "Running experiments... Output is streaming to terminal and logged to ${LOG_FILE}"

# Force Python into unbuffered mode so output shows up instantly while piping to tee
PYTHONUNBUFFERED=1 uv run run_experiments.py 2>&1 | tee "$LOG_FILE"

echo "========================================"
echo "Benchmarks complete!"
echo "Check the 'results' directory for the CSV and standard output logs."