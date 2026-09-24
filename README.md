# ringmaster

Composable long-context attention for distributed LLM training. Ulysses / Ring /
USP sequence parallelism that **wraps existing kernels** instead of shipping its
own — so it rides HuggingFace `kernels` Flash-Attention (FA2/FA3/FA4), torch SDPA,
and FlexAttention, with **no `flash_attn` pypi dependency**. Framework-neutral
core; thin adapters for axolotl, trl/accelerate, and raw PyTorch.

## Why

`ring-flash-attn` (the previous CP path) depends on the `flash_attn` pypi package,
which is no longer installed now that attention comes from HF `kernels`. ringmaster
takes two routes that avoid that dependency entirely:

- **Ulysses** wraps the model's *own* attention kernel with all-to-all on the head
  dim — works with any kernel (FA2/3/4/sdpa/flex), inherently load-balanced, the
  cheapest comm. Capped by KV-head count.
- **Ring** rides torch-native context parallel (`torch.distributed.tensor.experimental`)
  on the SDPA/FlexAttention backends — no head cap, scales linearly. LSE merging
  is handled inside aten, so we never reimplement a ring kernel.
- **USP** composes the two on a 2D mesh (Ulysses intra-node, Ring inter-node).

An **auto-selector** picks the `ulysses_size × ring_size` split from KV-head count,
world size, and intra-node topology.

## Install

```bash
pip install axolotl-ringmaster
```

The distribution is `axolotl-ringmaster` (the unrelated `ringmaster` name was taken
on PyPI); the import name is `ringmaster`.

## Quickstart

Requires **torch ≥ 2.11**. The framework adapters do this for you; the raw shape is:

```python
import ringmaster as rm

cfg = rm.RingmasterConfig(size=8, backend=rm.Backend.AUTO)
runtime = rm.setup(cfg, num_kv_heads=8, device_mesh=mesh)   # mesh has a "cp" dim
model.set_attn_implementation(runtime.attn_implementation)
wiring = rm.wire_recurrent_layers(model)                    # no-op for plain attention

from ringmaster.adapters.raw import context_parallel_region
for batch in loader:
    with context_parallel_region(batch) as shard:           # contiguous seq shard
        loss = model(**shard).loss
    loss.backward()
```

Pair CP with **FSDP2** (a `dp_shard × cp` mesh), not plain DDP — FSDP2's gradient
reduce-scatter sums the CP-partial gradients; DDP would mis-reduce. See
[docs/concepts.md](docs/concepts.md#device-mesh-and-fsdp2-composition).

## Documentation

- [docs/concepts.md](docs/concepts.md) — backends (Ulysses/Ring/USP), rotate
  methods, load balancing, device-mesh / FSDP2 composition, and when to use which.
- [docs/integration.md](docs/integration.md) — raw PyTorch, trl/accelerate
  (`CPSFTTrainer` + `ParallelismConfig` + `chunked_nll`), and axolotl (plugin + YAML).
- [docs/api.md](docs/api.md) — concise reference of the public API.
- [docs/models.md](docs/models.md) — plain attention, gated linear-attention
  (Qwen3.5/Next), Mamba2 hybrids (Nemotron-H), and the sm_120 shim.
- [examples/cp_fsdp2/](examples/cp_fsdp2/) — worked TRL + FSDP2 script and axolotl
  configs for all three model families, plus a CP-vs-non-CP benchmark table.

## Status

Requires **torch ≥ 2.11**.

| Phase | Feature | State |
|-------|---------|-------|
| 0 | Package scaffold, config, mesh, runtime, registry, loss | ✅ |
| 1 | Ulysses backend (wraps HF kernels FA2/3/4/sdpa) | ✅ GPU-validated — sdpa + flash parity, fwd+bwd |
| 2 | Ring — kernel-agnostic loop; **p2p (memory-optimal, fwd+bwd) + allgather (fast)** rotaters; `hf_kernels`/`torch_native`/`math` providers | ✅ GPU parity (both providers) + gloo fwd+bwd grad parity (p2p **and** allgather) |
| 3 | USP 2D composition (Ulysses × Ring) + auto-selector | ✅ **2×2 validated** (4 gloo ranks) + auto-selector |
| 4 | Mamba2 SSM CP (owned: `strategies/mamba.py`) + linear-attention state-passing | ✅ **mamba2 CP validated with the real mamba-ssm kernel (2 GPU)**; **gated-DeltaNet linear attention validated with the real flash-linear-attention kernel (2 GPU)**; cp_linear_scan + mamba2-vs-SSD (gloo); sliding-window via flash `window` |
| 5 | ALST memory | deferred to host framework (axolotl-native) |
| 6 | Profiling, trl/accelerate + multimodal adapters, **accelerate ParallelismConfig compat**, benchmark, docs, axolotl plugin (schema-merge tested) | ✅ |
| 7 | Custom collectives — fused Q/K/V all-to-all (MHA) ✅; **comm/compute overlap (prefetch) in p2p ring** ✅; symmetric-memory / Triton | partial (two items done) |
| v2 | packing / varlen | deferred |

## Benchmark (2× GPU, Ulysses, Llama hidden=2048/heads=16/layers=6, fwd+bwd)

| seq | non-CP (1 GPU) peak | CP (2 GPU) per-GPU peak | CP step |
|----:|----:|----:|----:|
| 16k | 8.4 GB | 4.6 GB | 440 ms |
| 65k | 31 GB | 16 GB | 2.1 s |
| 131k | 62 GB (10.1 s) | 31 GB | 6.3 s |
| 262k | **OOM** | 62 GB | 21 s |

CP halves per-GPU memory at every length (1/world sequence sharding) and reaches
2× the sequence (262k vs 131k) on the same hardware. `benchmarks/cp_vs_noncp.py`.

**CP is a memory/latency tool, not a throughput win.** It uses `world` GPUs, so
total GPU-time is higher than non-CP for a sequence that already fits on one GPU:
at 131k, non-CP = 10.1 GPU-s vs CP = 6.3 s × 2 = 12.6 GPU-s (~1.25×); at 16k the
overhead is ~2.4× (comm dominates). The overhead shrinks as sequence grows
(attention compute is superlinear, comm linear) but never breaks even. Use CP for
context that won't fit on one GPU, or to cut latency when you have spare GPUs; for
sequences that fit, DP is more GPU-efficient.

**p2p ring is the memory-optimal path** (only O(S/P) KV resident) with a proper
flash-style reverse-ring backward and DistFlashAttn-style comm/compute overlap
(prefetch the next KV block while computing the current). allgather is the
faster/higher-memory option (torchtitan-style). Both train (grad-parity validated).

**Remaining gap:** a live multi-GPU `axolotl train` run for correct *gradient*
semantics needs CP to be a non-DP mesh dim (via accelerate ParallelismConfig
`cp_size`, now supported by `ringmaster.parallelism.setup_from_parallelism_config`);
wiring axolotl's core to route through it (vs the legacy ring path) is the last
integration step.

Comm/memory are profilable: wrap any step in `rm.profile_comms()` / `rm.profile_memory()`
to get per-collective bytes+time and peak memory. The Ulysses leg fuses K/V into one
all-to-all (3 collectives/attn, not 4); the Ring leg offers `allgather` (fast, more
memory) vs `p2p` (O(S/P) KV resident) rotaters.

## Layout

```
ringmaster/
  config.py        RingmasterConfig (framework-neutral)
  compat.py        torch>=2.11 gate + experimental API isolation
  runtime.py       process-wide CP groups + active runtime
  mesh.py          resolve cp group(s), topology probe
  comm.py          SeqAllToAll4D (Ulysses all-to-all autograd; gloo fallback)
  profiling.py     profile_comms / profile_memory
  ring/
    loop.py        ring loop: allgather (autograd) + p2p (memory-frugal)
    kernels.py     block kernels: hf_kernels flash / aten flash (pluggable)
    merge.py       online-softmax LSE merge
  strategies/
    ulysses.py     AttentionInterface wrapper around HF kernels
    ring.py        register kernel-agnostic ring + impl resolution
    usp.py         2D composition + auto_select
    state_passing.py  Mamba2/linear exact CP linear scan
  shard.py         dense contiguous sequence sharding
  balancers.py     head_tail/contiguous index gen (Ring)
  loss.py          CP-correct token-weighted loss reduction
  multimodal.py    shard fused sequence after the encoder
  memory/          deferred to host framework (axolotl ALST)
  adapters/        raw PyTorch / trl+accelerate / hatchery-core entry points
  benchmarks/      profile_cp.py (comm + memory across seq lengths)
```

### Native recurrent context parallelism

Install `axolotl-ringmaster[fla]` for the native FLA GDN and KDA adapters.
`wire_recurrent_layers(model)` detects supported mixer contracts, installs
instance-local adapters, and returns a wiring object whose `restore()` method
undoes those changes. Install selected Hub kernels before wiring Mamba adapters;
the adapter captures the selected fused-kernel semantics at installation.
Unknown recurrent mixers fail during preflight. No model
architecture allowlist is required.

GDN and KDA currently require batch size one, contiguous unpacked shards, and
caching disabled. Transformers Mamba2 adapters exchange convolution halos and
scan states; packing metadata remains the caller's responsibility and packed
Mamba CP is rejected. Use a contiguous layout for all recurrent models.

Attention accepts dense causal sequences or globally right-padded batches through
the context manager. Left padding, holes, and arbitrary attention masks are
rejected. Packed attention uses global position boundaries; recurrent packing
combined with CP is not supported by these adapters.
