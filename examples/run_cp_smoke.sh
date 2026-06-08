#!/usr/bin/env bash
# e2e CP smoke on a real Qwen3.5-4B (gated-delta linear attention) over 2 GPUs,
# exercising both training stacks on top of ringmaster context parallelism:
#   1. raw TRL SFTTrainer  (qwen35_trl_cp_smoke.py)
#   2. axolotl plugin+FSDP2 (qwen35_axolotl_cp_fsdp.yaml)
# Both must take the native fla path (sm_120 TileLang warp-spec shim) and train
# with finite, decreasing loss. Exits non-zero if either path fails.
set -uo pipefail
cd "$(dirname "$0")"
NPROC="${NPROC:-2}"
fail=0

echo "==================== [1/2] TRL SFTTrainer + ringmaster CP ===================="
trl_log=$(mktemp)
accelerate launch --num_processes "$NPROC" --num_machines 1 --mixed_precision bf16 \
  qwen35_trl_cp_smoke.py 2>&1 | tee "$trl_log"
if grep -q "PATH=native" "$trl_log" && grep -q "\[trl_cp\] DONE" "$trl_log"; then
  echo "TRL smoke: PASS (native path, training completed)"
else
  echo "TRL smoke: FAIL"; fail=1
fi

echo "==================== [2/2] axolotl plugin + FSDP2 + ringmaster CP ============"
ax_log=$(mktemp)
axolotl train qwen35_axolotl_cp_fsdp.yaml 2>&1 | tee "$ax_log"
if grep -q "wired recurrent-layer CP" "$ax_log" && grep -q "Training completed" "$ax_log"; then
  echo "axolotl smoke: PASS (CP wired, training completed)"
else
  echo "axolotl smoke: FAIL"; fail=1
fi

echo "=============================================================================="
[ "$fail" -eq 0 ] && echo "CP SMOKE: ALL PASS" || echo "CP SMOKE: FAILURES"
exit "$fail"
