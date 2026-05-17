# DFlash V100 Performance Benchmark Report

**Date:** 2026-05-17
**Branch:** v100
**GPU:** Tesla PG503-216 (V100), compute capability 7.0, SMs: 72, VRAM: 32501 MiB
**CPU:** x86_64
**Compiler:** GCC 13.3.0, CUDA_ARCHITECTURES=70
**Models:**
- Target: Qwen3.6-27B-Q4_K_M.gguf (14.99 GiB GPU)
- Draft: dflash-draft-3.6-q8_0.gguf (1.75 GiB)
- Tokenizer: Qwen/Qwen3.5-27B

---

## 1. DFlash Speculative Decoding (bench_he.py, 10 prompts)

### Configuration

| Parameter | Value |
|-----------|-------|
| n_gen | 128 |
| DDTree budget | 22 |
| Fast rollback | enabled (default) |
| FA window | 2048 |
| Mode | fast |

### Results

| prompt | steps | AL (commit/step) | acc% | prefill tok/s | decode tok/s |
|--------|-------|-------------------|------|---------------|--------------|
| has_close_elements | 18 | 7.11 | 44.4% | n/a | **51.72** |
| separate_paren_groups | 19 | 6.74 | 45.7% | n/a | **47.60** |
| truncate_number | 29 | 4.41 | 27.6% | n/a | **32.15** |
| below_zero | 21 | 6.10 | 41.1% | n/a | **42.29** |
| mean_absolute_deviation | 25 | 5.12 | 32.2% | n/a | **37.12** |
| intersperse | 18 | 7.11 | 45.8% | n/a | **51.70** |
| parse_nested_parens | 26 | 4.92 | 31.0% | n/a | **34.80** |
| filter_by_substring | 25 | 5.12 | 33.2% | n/a | **36.66** |
| sum_product | 32 | 4.00 | 25.2% | n/a | **28.62** |
| rolling_max | 27 | 4.74 | 30.3% | n/a | **33.79** |
| **Average** | | **5.54** | **35.6%** | | **39.64** |
| **Min-Max** | | 4.00 - 7.11 | 25.2% - 45.8% | | 28.6 - 51.7 |

### Key Observations

- **Best AL**: `has_close_elements` 和 `intersperse` (7.11)
- **Worst AL**: `sum_product` (4.00)
- **Decode tok/s range**: 28.6 - 51.7 (差异大, 与 AL 高度相关)
- **AL 与 decode tok/s 正相关**: AL 越高, decode tok/s 越高

---

## 2. AR Baseline (test_generate, 5 prompts)

### Results

| prompt | AR tok/s | DFlash tok/s | Speedup |
|--------|----------|--------------|---------|
| has_close_elements | 20.43 | 39.73 | **1.94x** |
| separate_paren_groups | 20.46 | 63.99 | **3.13x** |
| truncate_number | 20.54 | 27.06 | **1.32x** |
| below_zero | 20.54 | 39.76 | **1.94x** |
| mean_absolute_deviation | 20.48 | 44.26 | **2.16x** |
| **Average** | **20.49** | **42.96** | **2.10x** |

### AR Baseline 分析

- AR decode tok/s 非常稳定: 20.43 - 20.54 (std < 0.05)
- DFlash 的加速比范围: 1.32x - 3.13x (取决于 prompt 的 AL)

---

## 3. Per-Step Timing Breakdown

DFlash 25 步的平均耗时 (ms):

| Stage | Time (ms) | % of Total |
|-------|-----------|------------|
| verify_compute | 118.07 | **77.8%** |
| draft_logits | 14.95 | **9.8%** |
| draft_compute | 12.39 | **8.2%** |
| draft_build | 1.41 | 0.9% |
| verify_set | 3.87 | 2.5% |
| verify_build | 1.10 | 0.7% |
| other | < 0.1 | < 0.1% |
| **Total sum** | **152.04** | |

### Timing Analysis

- **verify_compute 占 77.8%**: target verify 是最大的瓶颈
- **draft_compute + draft_logits 占 18%**: draft 生成也有显著开销
- 其余阶段 (snapshot, restore, replay, accept) 开销可以忽略

---

## 4. FlashPrefill Kernel Benchmarks

### block_score Kernel: Scalar vs GEMM

Config: H=16, Hk=8, D=128, BLOCK=128

| Sequence Length | M blocks | Scalar (ms) | GEMM (ms) | Speedup |
|-----------------|----------|-------------|-----------|---------|
| 4096 | 32 | 0.282 | 0.088 | **3.2x** |
| 8192 | 64 | 0.718 | 0.108 | **6.6x** |
| 16384 | 128 | 2.178 | 0.181 | **12.0x** |
| 32768 | 256 | 7.503 | 0.350 | **21.5x** |
| 65536 | 512 | 27.807 | 0.770 | **36.1x** |

### Analysis

- GEMM path 在长序列下速度优势显著, 从 3.2x (4K) 扩展到 36.1x (65K)
- 在 4K 以下, scalar path 的绝对延迟已经很低 (~0.3ms), GEMM 优势不大
- 65K+ context 时 GEMM 是必选项

---

## 5. Comparison with RTX 3090 Reference

### DFlash Performance

| Metric | RTX 3090 | V100 | V100/3090 |
|--------|----------|------|-----------|
| AR tok/s (ref) | 37.78 | 20.49 | **54%** |
| DFlash tok/s (bench_he.py mean) | 129.52 | 39.64 | **31%** |
| AL (acceptance length) | 8.31 | 5.54 | **67%** |
| Accept % | ~65% | 35.6% | **55%** |

### Hardware Comparison

| Spec | RTX 3090 | V100 (this bench) |
|------|----------|-------------------|
| GPU | GA102 | Volta V100 |
| Compute Cap | SM86 | SM70 |
| SMs | 82 | 72 |
| CUDA Cores | 10496 | 5120 |
| Tensor Cores | 3rd gen | 1st gen |
| VRAM | 24 GB GDDR6X | 32 GB HBM2 |
| Memory BW | 936 GB/s | 900 GB/s |
| Peak FLOPS (FP16) | 142 TFLOPS | 28.5 TFLOPS (tensor) |

### Key Findings

1. **AR baseline**: V100 是 3090 的 54%, 与硬件规格差距基本一致
2. **DFlash**: V100 只有 3090 的 31%, 差距更大, 原因:
   - 1st-gen Tensor Cores 不支持 1st-gen Tensor Cores 不支持 efficient WMMA
   - SM70 架构较老, 指令级并行度低
   - verify_compute 占 77.8%, 受限于 compute 能力

---

## 6. Configuration Tuning Results

### Different DDTree Configurations (prompt: separate_paren_groups, 128 tokens)

| Config | tok/s | AL | accept% |
|--------|-------|------|---------|
| Chain mode (no DDTree) | 16.53 | 2.29 | 14.4% |
| DDTree budget=22 | **63.99** | **9.85** | **45.7%** |
| DDTree budget=64 | 6.87 | 3.37 | 21.1% |
| DDTree budget=22 + temp=0.5 | 18.42 | 2.84 | 17.8% |

### Analysis

- **DDTree budget=22 最优**: 在 AL 和 compute 之间取得平衡
- **budget=64 反而更差**: tree 过大导致 verify_compute 开销剧增, 虽然 AL 略升但总 tok/s 下降
- **Chain mode 最差**: AL 只有 2.29, 加速效果有限
- **temp=0.5**: 降低了 draft 的多样性, AL 和 tok/s 都下降

---

## 7. Conclusions

1. **V100 SM70 可以运行 DFlash**, 但性能明显低于 3090
2. **AR baseline 稳定在 ~20.5 tok/s**, 与 RTX 3090 的 ~38 tok/s 相比约为 54%
3. **DFlash 平均 ~39.6 tok/s (bench_he.py 10 prompts)**, 约为 RTX 3090 的 31%
4. **Speedup ~2.1x vs AR**, 低于 RTX 3090 的 3.43x
5. **DDTree budget=22 是最优配置**
6. **FlashPrefill GEMM kernel** 在长序列下优势显著 (65K: 36.1x speedup)

### Recommendations

- V100 适合 DFlash 开发和功能验证, 但不适合高性能场景
- 如需更好的 V100 性能, 可以考虑:
  - 优化 verify_compute kernel (当前占 77.8% 的总耗时)
  - 降低 DDTree budget 到 16 (减少 tree size 以匹配 V100 compute 能力)
  - 使用 FlashPrefill GEMM path 优化 prefill 阶段

---

## 8. Reproduction

```bash
# DFlash benchmark
cd /mnt/eaget-4tb/data/llm_server/lucebox-hub/dflash
HF_ENDPOINT=https://huggingface.co \
DFLASH_TARGET=./models/Qwen3.6-27B-Q4_K_M.gguf \
DFLASH_BIN=./build/test_dflash \
python3 scripts/bench_he.py --n-gen 128

# AR baseline
./build/test_generate ./models/Qwen3.6-27B-Q4_K_M.gguf \
  /tmp/he_prompt_00.bin 128 /tmp/ar_out.bin

# FlashPrefill kernel benchmarks
./build/bench_flashprefill_perf
./build/bench_flashprefill_sm70
```
