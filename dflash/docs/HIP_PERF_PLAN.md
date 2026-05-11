# HIP perf — diagnosis + kernel-side optimization plan

_Drafted 2026-05-11 against `Luce-Org/lucebox-hub` post-#122 with HIP/ROCm
support landing in the upcoming PR. Numbers below are from the canonical
DFlash bench (Qwen3.6-27B-Q4_K_M + z-lab DFlash drafter,
`--fast-rollback --ddtree --ddtree-budget=22`, HE-style 128-tok prompt
md5 `4280413edc0b45c2b09e1a45f4f5ee60`, n_gen=256, warmup + 2 measurement
runs)._

## tl;dr

Lucebox HIP decode on gfx1100 (7900 XTX) runs **3.22× over AR** today —
within 6% of the README's CUDA RTX 3090 3.43× anchor. That's the headline
DFlash speedup. But the **absolute throughput is 50 tok/s on gfx1100 vs
~200 tok/s on hipfire's RDNA-native engine** on the same physical card
(same prompt, same target, same context). The 4× gap is **not** in
attention — it's in `mul_mat_q` for `q4_K` / `q4_0` / `q5_0`. The fix is
upstream in `ggml-cuda/mmq.cuh` + `mmvq.cuh`.

Tier 1 of this plan is **already empirically verified**: setting
`--ddtree-budget=8` instead of the default 22 on gfx1100 lifts decode
from 49.81 tok/s to 76.02 tok/s — a **53% speedup from a single config
flag**, no kernel work. Same flag is a -9% regression on gfx1201, so the
ship is arch-aware. Details in the Tier 1 section below.

**Tier 2 has been tested and FALSIFIED (2026-05-11).** The plan's original
hypothesis — that extending MMVQ template instantiations to cover
`ncols_dst ∈ {16, 23}` would route the DFlash verify path through the
BW-amortised GEMV kernel and recover most of the 4× gap — does not
hold. Three-arch A/B (gfx1100 / gfx1151 / gfx1201, 10-prompt HE bench,
default budget=22): MMVQ regresses **−42 to −69 % across all RDNA3+
silicon**. MMQ + WMMA on modern AMD wins decisively even at the
supposedly "wasteful" 32-wide tile occupancy of ne[1]=23 → 28% empty
columns. Full table + analysis in the Tier 2 section below. Pivot
recommendation: **Tier 3 (hipfire-style multi-row q4_K decode GEMV)
becomes the only kernel-side lever worth pursuing**.

This doc traces the rocprofv3 evidence, identifies the dispatch decisions
that route lucebox onto the slow path, and proposes a four-tier
optimization plan for the lucebox `llama.cpp-dflash-ggml` fork. Tier 2's
falsification is documented in full so future contributors don't repeat
the experiment.

## rocprofv3 top-10 hot kernels on gfx1100

Captured via `scripts/lucebox_kernel_atlas.py` (kernel-trace + summary +
ISA manifest) on the canonical DFlash bench above. Total profiled wall
~112s; per-kernel totals below cover all 256 generated tokens.

| Time   | Calls | Kernel | Notes |
|-------:|------:|---|---|
| 2076 ms | 1820 | `mul_mat_q<q4_K, 32, false>` | **target q4_K matmul, DDTree batch-tile 32** |
| 1247 ms | 8064 | `mul_mat_q<q4_0, 32, false>` | KV cache q4_0 matmul |
|  741 ms | 3456 | `Cijk_Alik_Bljk_SB_MT64x64x8_SN_1LDSB0_...` | rocBLAS strided batched GEMM (no WMMA) |
|  211 ms | 2304 | `mul_mat_q<q4_0, 16, false>` | smaller-tile MMQ |
|  205 ms | 1344 | `mul_mat_q<q5_0, 32, false>` | MMQ q5_0 |
|  130 ms |  420 | `Cijk_Alik_Bljk_HB_MT64x64x32_MI16x16x16x1_...` | rocBLAS GEMM **with** WMMA |
|  125 ms |  540 | `mul_mat_q<q4_K, 16, false>` | smaller-tile MMQ for q4_K |
|   72 ms | 1344 | `gated_delta_net_cuda<128, false, true, __half>` | DeltaNet hybrid path |
|   47 ms |  149 | `Cijk_..._MI16x16x16x1_...` | rocBLAS WMMA, second shape |
|   27 ms |  448 | `flash_attn_tile<256, 256, 32, 1, false>` | **FA tile — 0.5% of total** |

**~76% of GPU time is `mul_mat_q` variants. FlashAttention is 0.5%.**
The "huge tax" frame is correct, but it doesn't live in the missing
`flashprefill_kernels.hip.cu` (that path is short-prompt-cold here) — it
lives in `mmq.cuh`'s `q4_K` MMA path on RDNA3+.

## The dispatch trace — why DDTree always lands on MMQ

`ggml-cuda.cu:2294` decides MMVQ vs MMQ:

```cpp
bool use_mul_mat_vec_q = ggml_is_quantized(src0->type)
                        && ... && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
```

`MMVQ_MAX_BATCH_SIZE = 8` (`mmvq.cuh:3`).

DDTree budget=22 → speculation batch = 22 → `src1->ne[1] = 22 > 8` →
**always falls to MMQ.** MMQ uses WMMA on RDNA3+ (via
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` in `mma.cuh`) but the tile
shape is `32 × mmq_y × K` — designed for big-batch prefill. On a 22-wide
spec-verify batch it does 32 columns of work and discards 10 → **31%
wasted GPU compute on every speculation step**.

For the non-spec AR baseline (batch=1 decode), MMVQ kicks in correctly,
which is why the AR baseline at 28 tok/s is closer to hipfire's per-call
throughput than DFlash's 50 tok/s.

## Where MMVQ stops scaling — the second hard wall

`mmvq.cu:calc_nwarps()` and `calc_rows_per_block()` are explicitly tuned
for `ncols_dst ∈ {1..8}`. The fall-through for `ncols_dst ≥ 9` returns
`nwarps=1, rows_per_block=1` — i.e. no parallelism, one batch per warp,
launch-overhead-bound. Even if you bumped `MMVQ_MAX_BATCH_SIZE` to 32,
the kernel would behave badly because the per-arch
`MMVQ_PARAMETERS_RDNA3_0` / `RDNA4` cases gate on `ncols_dst == 1`
specifically (`mmvq.cu:326-345`).

So MMVQ as-shipped is not a drop-in fix for the 22-batch spec-verify
shape. It needs new instantiations.

## Four-tier optimization plan

### Tier 1 — Config-only (15 min, zero risk, **arch-specific**)

Try `--ddtree-budget=8` on the HIP backend. Routes spec-verify through
MMVQ instead of MMQ.

**Empirically validated 2026-05-11, n_gen=256 on the canonical HE bench
above, warmup + 2 measurement runs each:**

| Arch | Card | budget=22 (MMQ) | budget=8 (MMVQ) | Delta |
|---|---|---:|---:|---:|
| gfx1100 | 7900 XTX | 49.81 tok/s | **76.02 tok/s** | **+53%** |
| gfx1151 | Strix Halo iGPU | **34.78 tok/s** | 30.71 tok/s | -13% |
| gfx1201 | R9700 | **84.70 tok/s** | 77.23 tok/s | -9% |

The win is **gfx110x-only** (vanilla RDNA3 desktop dGPUs: 7900 XTX/XT,
7800 XT, 7700 XT/S, 7600). RDNA3.5 (Strix Halo gfx1151) and RDNA4
(gfx1201) both prefer MMQ at budget=22 — likely a combination of:

- RDNA4 has well-tuned MMQ tile shapes that make wasted columns of a
  batch-32 tile cheap proportionally.
- RDNA3.5 Strix Halo's LPDDR5X UMA (~270 GB/s vs 7900 XTX's 960 GB/s
  GDDR6) makes one-big-MMQ-launch's launch-amortization more valuable
  than tile-utilization. MMVQ's per-batch separate launches hurt UMA.

The dispatch analysis is correct on all three archs; the threshold is
just on the wrong side specifically for the desktop RDNA3 SKUs.

**Suggested ship**: arch-aware default in the daemon's CLI parsing or
`server.py` — set `--ddtree-budget=8` when running on gfx1100, gfx1101,
gfx1102 (desktop RDNA3 only). Keep 22 on gfx115x (RDNA3.5), gfx120x
(RDNA4), and CUDA. Single-PR change, zero kernel work, recovers most of
the gfx110x-specific gap.

```bash
./test_dflash $T $D prompt.bin 256 out.bin --fast-rollback --ddtree --ddtree-budget=22  # current default
./test_dflash $T $D prompt.bin 256 out.bin --fast-rollback --ddtree --ddtree-budget=8   # MMVQ-routed
```

### Tier 2 — Extend MMVQ template instantiations (TESTED 2026-05-11, FALSIFIED)

**The original hypothesis**: extend the MMVQ switch from `ncols_dst ∈
{1..8}` to also include the real DFlash verify shapes (`ne[1] = 16` for
chain mode = `DFLASH27B_DRAFT_BLOCK_SIZE`, `ne[1] = 23` for default
DDTree `--ddtree-budget=22` = `1 + budget`). Add a per-(type, cc, batch)
gate `mmvq_is_supported_batch()` at the dispatcher so RDNA3+ q4_K routes
to the BW-amortised vec_dot kernel instead of MMQ's 32-wide tile that
wastes ~28 % of ALU at ne[1]=23. Projected: **+50-100 % decode tok/s
on gfx1100**, cutting the dominant `mul_mat_q<q4_K, 32>` cost in half.

**Implementation** (research artifact at
`Kaden-Schutt/llama.cpp-dflash-ggml@feat/mmvq-rdna3-batch16`, commit
`002db52`, default-off behind `GGML_MMVQ_NO_EXTENDED=1` opt-out):

- New `mmvq_is_supported_batch(ggml_type, int cc, int batch)` returning
  true for `batch ∈ [1, 8]` unconditionally plus
  `batch ∈ {16, 23}` for q4_K on RDNA3 / RDNA4 (`GGML_CUDA_CC_IS_RDNA3`
  / `_IS_RDNA4`).
- New template instantiations at `case 16:` and `case 23:` in
  `mul_mat_vec_q_switch_ncols_dst<type>()`, with the existing
  `nwarps=1, rows_per_block=1` shape for non-`MMVQ_PARAMETERS_GENERIC`
  tables. Each new value adds one compile unit per quant type; bench
  binary size unchanged within rounding.
- Bumped assert ceiling `MMVQ_MAX_BATCH_SIZE_EXTENDED = 23`.
- Replaced the two host-side `src1->ne[1] <= MMVQ_MAX_BATCH_SIZE`
  gates at `ggml-cuda.cu:2294,2337` with the new
  `mmvq_is_supported_batch(...)` call.

**Bench A/B** — single binary per arch, `GGML_MMVQ_NO_EXTENDED=1` for
baseline cell vs unset for tier2 cell, byte-identical 10-prompt
HumanEval set (`bench_he.py`, `--n-gen 256 --skip-tokenize`),
Qwen3.6-27B-Q4_K_M target + matched z-lab/Qwen3.6-27B-DFlash drafter,
ROCm 7.2.2:

| GPU | Arch | Budget | baseline (MMQ) AL / tok/s | tier2 (MMVQ) AL / tok/s | Δ |
|---|---|---:|---:|---:|---:|
| 7900 XTX (k9lin)        | gfx1100 (RDNA3)   | **22** | 7.57 / **40.91** | 7.00 / 23.46 | **−42.7 %** |
| 7900 XTX (k9lin)        | gfx1100 (RDNA3)   | 8      | 5.25 / 62.36     | 5.55 / 65.72 | +5.4 % (noise, paths ≡) |
| R9700 (hiptrx)          | gfx1201 (RDNA4)   | **22** | 8.26 / **77.53** | 7.69 / 24.13 | **−68.9 %** |
| R9700 (hiptrx)          | gfx1201 (RDNA4)   | 8      | 5.51 / 64.88     | 6.05 / 70.77 | +9.1 % (noise) |
| Strix Halo iGPU (hipx)  | gfx1151 (RDNA3.5) | **22** | 6.66 / **26.36** | 8.24 / 14.79 | **−43.9 %** |
| Strix Halo iGPU (hipx)  | gfx1151 (RDNA3.5) | 8      | 5.56 / 28.73     | 5.17 / 26.70 | −7.1 % (noise; AL drift) |

**Tier 2 falsified on all three RDNA3+ archs at the default workload**.
The budget=8 cells are bench-noise null: at ne[1]=9 the new gate also
routes to MMQ (9 ∉ {1..8, 16, 23}), so the cells exercise identical
kernels modulo FP-reduction-order drift.

**Why the hypothesis was wrong**:

1. **WMMA beats BW amortisation on modern AMD even with wasted lanes.**
   MMQ at ne[1]=23 uses a 32-wide tile (28 % of columns sit idle), but
   the WMMA matrix cores process 16×16 tiles at ~four bf16-FMA-per-cycle
   per lane. MMVQ's `vec_dot_q4_K_q8_1` is scalar `v_dot4` (RDNA3) /
   scalar fp16 FMA (RDNA1/2) — no matrix-core throughput. The
   ALU-density gap dwarfs the tile-occupancy loss.
2. **MMVQ re-reads activations per `ncols_dst`.** Inside the K-block
   loop the kernel does
   `tmp[j][i] += vec_dot(vx, &y[j*stride_col_y + kby], ...)` for
   `j ∈ [0, ncols_dst)`. At ncols_dst=23 that's 23 redundant activation
   loads per K-block, all hitting GDDR6 (RDNA3 dGPU) / LPDDR5X (Strix
   Halo). MMQ stages activations into LDS once per tile and amortises
   across `tile_M × tile_N = 16 × 32 = 512` output positions.
3. **Per-thread accumulator pressure.** `float tmp[23][1]` on the stack
   is 23 live VGPRs per thread, plus the vec_dot working set. The
   compiler doesn't spill on gfx1100, but the larger live-set reduces
   wave-occupancy on RDNA3.5 and RDNA4 enough to compound with the BW
   issue.
4. **FP non-associativity.** MMQ tile-summed accumulation and MMVQ
   K-block + warp_reduce_sum produce different bit-patterns for the
   same logits. That shifts argmax by enough on a fraction of tokens
   that AL drifts ±7 % per arch, on top of the kernel slowdown.

**Negative-result class**. This is the fifth synth-win → prod-falsify
cycle on RDNA3+ for kernel-level decode optimisations under ROCm 7.2.x
(hipfire's prior catalogue: FP8 dot4 GEMV on gfx1201, gfx12 FP8 WMMA on
HFP4G32, MFP4 MoE all-FP4, gemv graph cache PR3). The pattern: per-shape
microbenches show 1.5-2× wins; the same change loses 5-70 % in
end-to-end production because cross-kernel L2 state, scheduler
ordering, and WMMA-vs-scalar ALU density only become legible at the
full forward pass. **Only launch-reduction levers (β + graph capture,
fusion) cross zero in production on this codebase + ROCm version**.

**Disposition**:

- Keep the `feat/mmvq-rdna3-batch16` branch as a research artifact;
  default-off behind `GGML_MMVQ_NO_EXTENDED` so a non-set env still
  triggers it for reproduction. **Do not open as a perf PR**.
- Pivot the kernel-side roadmap to Tier 3 (next section): a multi-row
  q4_K decode GEMV that processes R=4-8 rows per warp with shared
  activation register state, the hipfire pattern that consistently
  ships on RDNA3+ without the wasted-tile / per-row-reread tradeoff
  that killed Tier 2.

### Tier 3 — Multi-row decode GEMV à la hipfire (1-2 weeks)

Hipfire's `kernels/src/gemv_hfq4g256_multirow.gfx1100.hip` processes
R=2/4/8 output rows per warp, sharing the X (activation) register state
across rows. For a wide decode batch (DDTree budget=22-32) this is
exactly the right shape:

- One warp processes 4 output cols × 22 batches → 88 dot products with
  one X-load
- Register pressure: ~38 VGPRs for R=4 multirow (hipfire-measured on
  gfx1100), still 16 waves/CU occupancy
- vs MMQ's tile-based approach which does 32 cols × mmq_y rows but
  burns more shared memory and launches more thread blocks

To port to ggml's q4_K shape, the kernel needs:
- The q4_K block layout reader (super-blocks of 256 with 6-bit scales +
  6-bit mins per sub-block of 32)
- A wave32 fast path using `v_dot4_i32_i8` for non-WMMA inner loops on
  gfx1010/1030, and `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` for
  gfx1100+
- Register-packed batch dimension to avoid LDS-staging for activations

Reference patterns in hipfire's tree (not directly portable, but the
tile/loop structure transfers):
- `crates/rdna-compute/src/dispatch.rs:601-613` — the per-arch ROCm-min +
  WMMA-builtin routing table
- `crates/rdna-compute/src/kernels.rs:multirow_*` — the R-selection logic
- `kernels/src/gemv_hfq4g256_multirow.gfx1100.hip` — the actual kernel

Expected impact: **2-3× on DFlash spec-verify**, bringing q4_K decode
within ~20% of hipfire on the same hardware.

### Tier 4 — gfx1010 / gfx1030 scalar-fallback score kernel (3-5 days)

Orthogonal to the q4_K decode work. Required to unblock PFlash on
RDNA1/RDNA2 cards where today the score-blocks kernel hangs (gfx1010 —
missing `v_dot4`) or runs ~7× slower than Strix Halo (gfx1030 — uses
SDWA fallback but no WMMA available).

Pattern: hipfire's `kernels/src/gemv_hfq4g256_multirow.gfx1010.hip`.
Wave32 RDNA1, scalar fp16 accumulation, no WMMA dependency.

## Ranked priority (revised post-Tier-2 falsification)

1. **Tier 1 today** — needs only an arch-aware default change in
   `bench_he.py` / `run.py` / `server.py`, gives a free 53 % bump on
   gfx1100 (`--ddtree-budget=8`). Zero kernel work, lowest risk, ships
   first.
2. ~~Tier 2~~ **— removed**. Bench data above shows MMVQ at extended
   `ncols_dst` regresses 42-69 % across all RDNA3+ silicon at the
   default workload. The branch survives as a documented research
   artifact (default-off env-gated) so future investigators can
   reproduce the negative result without re-implementing the path.
3. **Tier 3 — primary kernel-side lever**. The real engineering work
   that consistently ships on RDNA3+ in hipfire: multi-row q4_K decode
   GEMV with shared activation register state. Sidesteps both failure
   modes of Tier 2 — no wasted-tile waste (R=4-8 rows is the actual
   live output budget) and no per-`ncols_dst` activation re-read (the
   batch dimension is register-packed, not loop-extended). Lands as
   PR-C below. **~1-2 weeks of focused work; projected +1.5-3× on
   DFlash spec-verify on gfx1100/1201**.
4. **Tier 4 in parallel** — unlocks RDNA1/RDNA2 PFlash entirely. No
   overlap with Tier 3; can ship in any order.

Path B (rocWMMA port of `flashprefill_kernels.cu`) addresses the PFlash
*prefill* tax (compress + target_prefill on long ctx). It is **orthogonal
to this plan** — it helps long-context TTFT, not decode tok/s. Both
should ship.

## Sequence of PRs against `Luce-Org/llama.cpp-dflash-ggml`

1. **PR-A**: arch-aware `--ddtree-budget` default (Tier 1, daemon-side
   only, no submodule change).
2. **PR-C**: multi-row decode GEMV for q4_K on RDNA3+ (Tier 3, biggest
   remaining payoff). Lands a new
   `mul_mat_vec_q_multirow_rdna_<arch>.cu` template alongside the
   existing MMVQ kernel; dispatched from `ggml_cuda_mul_mat` at
   `ne[1] ≤ 8` (the regime where the existing MMVQ already wins) to
   capture R-row sharing without touching the > 8 dispatcher (which
   Tier 2 proved is MMQ's territory).
3. **PR-D**: gfx1010/1030 scalar-fallback score kernel (Tier 4).
4. **PR-E (separate, against `lucebox-hub`)**: rocWMMA port of
   `flashprefill_kernels.cu` → `flashprefill_kernels.hip.cu` (Path B,
   prefill tax).

Each PR is independent, separately bench-able, separately revertible.
Tier 2's original PR-A (extend MMVQ to ncols_dst ≤ 32) and PR-B
(generalise to all common quants) have been **dropped** from this
sequence; rationale in the Tier 2 section above.

## Validation per PR

- Bit-identical token output vs CUDA baseline (gfx1100 vs RTX 30/40-series
  on the same prompt). Lucebox already has `test_vs_oracle` for this.
- DFlash 3-tier coherence smoke (Path-A attractor / 3gram density /
  EOS-immediate) — port from hipfire `crates/hipfire-detect/` as a small
  Python script.
- NIAH retrieval at 8K / 32K / 64K / 128K — already verified end-to-end
  on the HIP support PR.
- Prompt-md5 disciplined bench: warmup + ≥2 measurement runs, fresh
  binary, prompt md5 logged. See lucebox's existing `bench_he.py` setup.
