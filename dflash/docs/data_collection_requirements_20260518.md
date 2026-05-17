# DFlash Training Data Collection Requirements

## Environment Setup

### Required Dependencies

```bash
# Python 3.10+
pip install torch>=2.0.0 transformers>=4.40.0 datasets loguru tqdm safetensors
```

### Model Requirements

The data collection script requires a **HuggingFace format** model, not GGUF.

#### Option 1: Download HuggingFace Model (Recommended)

```bash
# Convert from GGUF to HF (requires llama.cpp conversion)
# Or download directly from HuggingFace
huggingface-cli download Qwen/Qwen3.6-27B \
    --local-dir models/Qwen3.6-27B-HF \
    --local-dir-use-symlinks False
```

#### Option 2: Use Existing Model Path

```bash
python scripts/run_collect_draft_data.py \
    --model-path /path/to/Qwen3.6-27B-HF \
    --pilot
```

### GPU Requirements

- **Minimum**: 24GB VRAM (e.g., RTX 3090, RTX 4090)
- **Recommended**: 32GB+ VRAM (e.g., V100, A100)
- **DType**: BF16 recommended for memory efficiency

## Data Collection Modes

### 1. Pilot Mode (Quick Test)

```bash
# Collect ~700 samples (100 per dataset)
python scripts/run_collect_draft_data.py --pilot

# Expected output: models/training_data/draft_training_qwen36_pilot_YYYYMMDD_HHMMSS.pt
# Expected time: ~30-60 minutes
```

### 2. Full Collection

```bash
# Collect ~5200 samples (default configuration)
python scripts/run_collect_draft_data.py

# Or specify custom sample count
python scripts/run_collect_draft_data.py --num-samples 10000
```

### 3. Specific Datasets

```bash
# Collect only from specific datasets
python scripts/run_collect_draft_data.py \
    --datasets humaneval math500 gsm8k \
    --num-samples 5000
```

### 4. Merge Existing Collections

```bash
# Merge multiple data files
python scripts/run_collect_draft_data.py \
    --merge models/training_data/*.pt
```

## Dataset Coverage

The collection pipeline covers the following domains:

| Dataset | Domain | Samples | Field |
|---------|--------|---------|-------|
| openai/open-eval-extra | Code (HumanEval) | 500 | problem |
| mbpp | Code (MBPP) | 500 | text |
| HuggingFaceH4/MATH-500 | Math | 500 | problem |
| openai/gsm8k | Math (GSM8K) | 1000 | question |
| ShareGPT_V3 | Conversation | 2000 | conversations |
| LongProcedureQA | Long Context | 200 | input |
| alpaca-cleaned | Long Context | 500 | instruction |

## Output Format

Each sample contains:

```python
{
    "input_ids": Tensor[seq_len],      # Input tokens
    "input_len": int,                   # Input length
    "target_ids": Tensor[gen_len],      # Generated tokens
    "prompt_hidden": {                  # Hidden states at last input token
        "layer_1": Tensor[1, 1, 5120],
        "layer_16": Tensor[1, 1, 5120],
        "layer_31": Tensor[1, 1, 5120],
        "layer_46": Tensor[1, 1, 5120],
        "layer_60": Tensor[1, 1, 5120],
    },
    "gen_hidden": {                     # Hidden states during generation
        "layer_1": Tensor[gen_len, 5120],
        # ... (same structure for all layers)
    },
    "source": str,                      # Dataset name
    "prompt": str,                      # Truncated prompt
}
```

## Validation

The script automatically validates:
- **Dimension correctness**: Hidden states must have shape `[seq_len, 5120]`
- **No NaN/Inf values**: Invalid samples are counted but not removed
- **Complete pairs**: Each sample has both input and target tokens

## Troubleshooting

### Out of Memory

```bash
# Reduce max_length
python scripts/run_collect_draft_data.py \
    --max-length 1024 \
    --pilot
```

### Dataset Load Errors

Some datasets may require authentication or have changed structure:

```bash
# Use specific datasets only
python scripts/run_collect_draft_data.py \
    --datasets humaneval math500
```

### Model Download Issues

If HuggingFace download fails, use a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/run_collect_draft_data.py
```

## Next Steps After Collection

1. **Validate data quality**:
   ```bash
   python scripts/validate_training_data.py \
       --data models/training_data/draft_training_qwen36_*.pt
   ```

2. **Train draft model**:
   ```bash
   python scripts/train_draft_qwen36.py \
       --data models/training_data/draft_training_qwen36_*.pt \
       --output models/draft/dflash-draft-3.6-trained.pt
   ```

3. **Convert to GGUF**:
   ```bash
   python scripts/convert_dflash_to_gguf.py \
       --model models/draft/dflash-draft-3.6-trained.pt \
       --output models/draft/dflash-draft-3.6-trained-Q8_0.gguf
   ```

## Resource Estimates

| Mode | Samples | Time | Storage |
|------|---------|------|---------|
| Pilot | ~700 | 30-60 min | ~2 GB |
| Full (default) | ~5200 | 4-6 hours | ~15 GB |
| Full (10K/ds) | ~36000 | 24-30 hours | ~100 GB |
