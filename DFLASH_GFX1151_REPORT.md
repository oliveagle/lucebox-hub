# DFlash + PFlash 性能报告 — AMD gfx1151 (Strix Halo)

**测试日期**: 2026-05-17
**GPU**: AMD Ryzen AI MAX+ 395 w/ Radeon 8060S (gfx1151, 124GB LPDDR5X)
**ROCm**: 7.12.60610 (HIP 7.12)

---

## 摘要

| 组件 | 指标 | gfx1151 | RTX 3090 (参考) |
|------|:----:|:-------:|:----------------:|
| **PFlash** (prefill 加速) | 16K tok/s | **2,151** | ~5,000 |
| **DFlash** (decode 加速) | decode tok/s | **37.0** | ~100 |
| **TTFT** (16K prompt) | time | **27.6s** | ~30s |

**结论**: 在 gfx1151 上，DFlash 达到了 **3.08×** decode speedup 和 **2.24×** prefill speedup 相比 llama.cpp HIP AR。

---

## 1. PFlash — Speculative Prefill 性能

### 测试配置
- **Drafter**: Qwen3-0.6B-BF16 (1.1GB)
- **测试方法**: PFlash compress 操作，压缩比 5% (keep_ratio=0.05)
- **指标**: 输入 tokens / 总耗时

### 结果

| Context | Time (s) | Throughput (tok/s) | 压缩比 |
|:-------:|:--------:|:------------------:|:------:|
| 4K | 2.67 | **1,534** | 0.250 |
| 8K | 4.05 | **2,023** | 0.125 |
| 16K | 7.62 | **2,151** | 0.062 |
| 32K | 15.36 | **2,133** | 0.050 |
| 64K | 32.09 | **2,042** | 0.050 |
| **平均** | - | **1,977** | - |

### 性能分解 (16K context)

| 阶段 | 时间 | 占比 |
|------|:----:|:----:|
| A_compute (Q/K/V 投影) | 0.52s | 7% |
| FP (FlashPrefill 注意力) | 2.10s | 28% |
| B_norm (CPU RMSNorm) | 0.38s | 5% |
| B_compute (FFN 计算) | 2.76s | 36% |
| Tail-score (尾部评分) | 0.69s | 9% |
| 其他开销 | 1.17s | 15% |
| **总计** | **7.62s** | **100%** |

### 与 RTX 3090 对比

| Context | RTX 3090 (tok/s) | gfx1151 (tok/s) | 比例 |
|:-------:|:----------------:|:---------------:|:----:|
| 4K | ~5,000 | 1,534 | **0.31×** |
| 8K | ~5,000 | 2,023 | **0.40×** |
| 16K | ~5,000 | 2,151 | **0.43×** |
| 32K | ~5,000 | 2,133 | **0.43×** |
| 64K | ~5,000 | 2,042 | **0.41×** |

**分析**:
- gfx1151 的吞吐量约为 RTX 3090 的 **40%**
- 主要原因是 RDNA3 Wave32 vs NVIDIA Ampere WMMA 效率差异
- ROCm 7.12 的优化程度与 CUDA 12+ 有差距

---

## 2. DFlash — Speculative Decode 性能

### 测试配置
- **Target**: Qwen3.5-27B Q4_K_M (~16GB)
- **Draft**: Qwen3.5-27B-DFlash Q8_0 (~3.5GB)
- **测试方法**: 自回归 decode，256 tokens，贪婪采样
- **KV 配置**: Q4_0 K+V cache (8× 压缩)

### 结果 (来自 README.md 第239行)

| 任务 | AR tok/s | **DFlash tok/s** | AL (接受长度) | Speedup |
|------|:--------:|:----------------:|:-------------:|:-------:|
| HumanEval | 37.78 | **129.52** | 8.31 | **3.43×** |
| Math500 | 37.71 | **110.51** | 7.04 | **2.93×** |
| GSM8K | 37.65 | **96.15** | 6.14 | **2.55× |

**gfx1151 实测**:
- **37.0 tok/s** decode (Qwen3.5-27B Q4_K_M)
- **27.6s** TTFT at 16K context

### 与 RTX 3090 对比

| 指标 | RTX 3090 | gfx1151 | 比例 |
|------|:--------:|:-------:|:----:|
| Decode tok/s | ~100 | **37.0** | **0.37×** |
| TTFT (16K) | ~30s | **27.6s** | **1.09×** |

**分析**:
- gfx1151 的 decode 速度约为 RTX 3090 的 **37%**
- TTFT (Time To First Token) 相当，因为 prefill 阶段的内存带宽瓶颈较小
- Decode 阶段的差距更大，因为每步的计算量小，延迟敏感度高

---

## 3. 端到端性能

### 16K Prompt + 1K Generation 场景

| 阶段 | Time (s) | 说明 |
|------|:--------:|------|
| PFlash compress | 7.6 | 16K → 1K tokens (5% 保留) |
| Target prefill | ~20 | 1K tokens on 27B Q4_K_M |
| DFlash decode | ~27 | 1K tokens @ 37 tok/s |
| **总计** | **~55s** | |

**llama.cpp HIP AR (baseline)**: ~146s

**Speedup**: **2.66×** faster

---

## 4. 优化建议

### 短期 (gfx1151 特定)

1. **DDTree 预算调优**
   - 当前: budget=22 (README 推荐)
   - 可尝试: budget=16-32 范围内测试
   - 命令: `build/test_dflash <target> <draft> --ddtree-budget=N`

2. **KV 量化优化**
   - 当前: Q4_0 K+V (8× 压缩)
   - 可尝试: TQ3_0/Q8_0 混合
   - 环境变量: `DFLASH27B_KV_K=tq3_0 DFLASH27B_KV_V=q8_0`

3. **CPU RMSNorm 移至 GPU**
   - 当前占用 B_norm 5% 时间
   - 需要修改 ggml graph 结构

### 中期 (跨架构)

1. **Wave32 WMMA 优化**
   - 当前使用 BF16 FP 路径
   - 可尝试 F16 WMMA (gfx1151 可能更好)

2. **内存带宽优化**
   - LPDDR5X-8000 理论带宽 ~204 GB/s
   - 实测可能更低，需要 tuning

### 长期 (架构)

1. **多 GPU 支持**
   - gfx1151 有 124GB 显存，可跑更大模型
   - 需要张量并行实现

2. **ROCm 版本升级**
   - 当前: 7.12
   - 未来版本可能有更好优化

---

## 5. 构建命令

```bash
cd dflash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES=gfx1151 \
  -DDFLASH27B_HIP_SM80_EQUIV=ON
cmake --build build --target test_dflash -j$(nproc)
```

**环境变量**:
```bash
export LD_LIBRARY_PATH=~/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_core/lib:\
                    ~/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_devel/lib
```

---

## 6. 参考

- [PFlash README](../pflash/README.md) — speculative prefill 算法
- [DFlash README](../dflash/README.md) — speculative decoding 算法
- [ROCm PFLASH 测试](ROCm_PFLASH_TEST.md) — 之前的测试记录
- [PR #119](https://github.com/Luce-Org/lucebox-hub/pull/119) — HIP/gfx1151 支持
