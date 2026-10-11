#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATASETS=MATH-500,AMC23,AIME24,HMMT25,Minerva,Olympiad-Bench,GSM8K,HumanEvalPlus,MBPPPlus,LiveCodeBench,BigCodeBench-Hard
exec bash "$REPO_ROOT/evaluation/run_eval.sh" --datasets "$DATASETS" "$@"
