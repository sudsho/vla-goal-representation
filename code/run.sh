#!/bin/bash
# usage: bash code/run.sh <plan> <outname> <nworkers> [extra rollout.py args]
# Each worker auto-restarts on crash (resume skips finished episodes).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=2 USE_FLAX=0 USE_TF=0 LIBERO_CONFIG_PATH="data/libero_cfg" TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
plan=$1; out=$2; nw=${3:-1}; shift 3
mkdir -p data/runs/logs
for w in $(seq 0 $((nw-1))); do
  (
    for try in $(seq 1 40); do
      .venv/Scripts/python code/rollout.py --ckpt data/ckpt/openvla-7b-finetuned-libero-goal \
        --plan plans/$plan.jsonl --out data/runs/$out --worker $w --nworkers $nw "$@" >> data/runs/logs/${out}_w$w.log 2>&1 && break
      echo "=== RESTART $try $(date) ===" >> data/runs/logs/${out}_w$w.log; sleep 15
    done
  ) &
  sleep 60
done
wait
