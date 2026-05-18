# Qwen3.6-27B TriAttention Benchmark Plan

**Date**: 2026-05-18
**Task**: lucebox-hub-gfx1151-z0k.1
**Model**: Qwen3.6-27B
**GPU**: AMD Radeon 8060S (gfx1151, ROCm 7.2.3)
**Inference Engine**: vLLM + TriAttention

## Status

⚠️ **BLOCKED**: PyTorch/vLLM incompatibility with AMD GPU

### Current Environment
- PyTorch: 2.11.0+cu130 (CUDA-compiled, cannot use AMD GPU)
- vLLM: 0.21.0 (CUDA-dependent)
- ROCm: 7.2.3 installed at /opt/rocm
- GPU: AMD Radeon 8060S (34% VRAM usage, ~21GB available)

### Issues Encountered

1. **vLLM CUDA Dependency**: Current vLLM installation requires NVIDIA CUDA, incompatible with AMD GPU
2. **llama.cpp Build Error**: HIP/CUDA header conflicts preventing successful build
3. **TriAttention ROCm Support**: Community port available but requires proper build environment

## Workarounds Being Explored

### Option 1: Install ROCm-compiled PyTorch and vLLM
```bash
# Uninstall CUDA PyTorch
pip uninstall torch torchvision torchaudio

# Install ROCm PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Check vLLM ROCm support
# vLLM has limited ROCm support - may need to build from source
```

### Option 2: Use llama.cpp with TriAttention Community Port
- **Project**: [triattention-ggml](https://github.com/domvox/triattention-ggml)
- **Maintainer**: @domvox
- **Status**: HIP/ROCm support for AMD GPUs
- **Issue**: Build failing with HIP/CUDA header conflicts

### Option 3: CPU Fallback Testing
- Test with smaller model (Qwen3-8B) on CPU
- Establish benchmark methodology
- Document for future GPU testing

## Benchmark Plan (Once Environment is Ready)

### Test Scenarios

| Scenario | Context Length | Output Length | Description |
|----------|----------------|---------------|-------------|
| Short | 4,096 | 512 | Standard chat |
| Medium | 8,192 | 512 | Document QA |
| Long | 16,384 | 512 | Long context |

### Configuration

```bash
# TriAttention Settings
KV_BUDGET=2048  # or 3072
DIVIDE_LENGTH=128
WINDOW_SIZE=128
STATS_PATH=./submodules/triattention/triattention/vllm/stats/qwen3_6_27b_stats.pt

# vLLM Settings
--dtype bfloat16
--max-model-len 32768
--enforce-eager
--trust-remote-code
--no-enable-prefix-caching
--gpu-memory-utilization 0.85
```

### Metrics to Collect

| Metric | Unit | Description |
|--------|------|-------------|
| Prefill Speed | tokens/s | Prompt processing |
| Decode Speed | tokens/s | Token generation |
| First Token Latency | ms | Time to first token |
| Total Time | s | End-to-end time |
| KV Memory | GB | KV cache usage |
| Accept Rate | % | TriAttention effectiveness |

### Test Prompts

```python
# Short context (4K)
prompt_short = "Write a detailed explanation of how transformer models work, including self-attention mechanisms, positional encoding, and the transformer architecture. Include examples and diagrams where applicable."

# Medium context (8K)
prompt_medium = "Explain the history of artificial intelligence from the 1950s to present day, covering: 1) Early AI research and the Dartmouth conference, 2) Expert systems and knowledge representation, 3) Machine learning revolutions, 4) Deep learning breakthroughs, 5) Modern large language models, 6) Current challenges and future directions. Provide specific years, key researchers, and pivotal papers for each era."

# Long context (16K)
prompt_long = "[Comprehensive technical document spanning multiple pages...]"
```

## Next Steps

1. **Immediate**: Fix PyTorch/vLLM for ROCm compatibility
   - Uninstall CUDA PyTorch
   - Install ROCm PyTorch
   - Check vLLM ROCm support status
   - Consider building vLLM from source with ROCm

2. **Alternative**: Complete llama.cpp build
   - Fix HIP/CUDA header conflicts
   - Build llama-server with ROCm
   - Integrate TriAttention community port

3. **Fallback**: CPU testing with smaller model
   - Test Qwen3-8B on CPU
   - Validate benchmark methodology
   - Prepare for GPU testing once environment is ready

## Resources

- **Model Path**: `/mnt/eaget-4tb/modelscope_models/Qwen/Qwen3___6-27B`
- **TriAttention Stats**: `submodules/triattention/triattention/vllm/stats/qwen3_6_27b_stats.pt`
- **TriAttention Docs**: `submodules/triattention/README.md`
- **ROCm Version**: 7.2.3
- **GPU**: AMD Radeon 8060S (gfx1151)

## References

- [TriAttention Paper](https://arxiv.org/abs/2604.04921)
- [TriAttention README](submodules/triattention/README.md)
- [triattention-ggml (AMD GPU support)](https://github.com/domvox/triattention-ggml)
- [vLLM ROCm Support](https://github.com/vllm-project/vllm)
- [Project CLAUDE.md](CLAUDE.md)
