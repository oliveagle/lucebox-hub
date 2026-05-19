# tria_k_pre_rope Pointer Corruption Root Cause Investigation

## Executive Summary

The `cache.tria_k_pre_rope` pointer corruption issue manifests when the pointer changes from a valid heap address (e.g., `0x59cac61925e0`) to an invalid stack address (e.g., `0x7ffd2dd32f30`) between allocation and use in `build_full_attn_block()`.

## Current Workaround

Both call sites to `build_full_attn_block()` pass `nullptr` for `tria_k_pre_rope`:
- Line 1017 in `build_single_layer()`
- Line 1133 in `build_qwen35_graph()`

This prevents TriAttention pre-RoPE K capture but allows DFlash to function.

## Investigation Findings

### 1. Allocation Path (Verified Working)

**File:** `dflash/src/qwen35/qwen35_target_graph.cpp`

**Allocation (lines 193-208):**
```cpp
out.tria_k_pre_rope = ggml_new_tensor_3d(out.base_ctx, GGML_TYPE_BF16,
                                          head_dim, max_ctx_alloc, w.n_head_kv);
```

**Backend allocation (line 220):**
```cpp
out.base_buf = ggml_backend_alloc_ctx_tensors(out.base_ctx, backend);
```

**Logging (lines 221-223):**
```cpp
fprintf(stderr, "[TriAttention] base_buf=%p tria_k_pre_rope=%p data=%p\n",
        (void*)out.base_buf, (void*)out.tria_k_pre_rope,
        out.tria_k_pre_rope ? out.tria_k_pre_rope->data : nullptr);
```

The allocation is successful. The log shows valid pointers:
- `tria_k_pre_rope=0x59cac61925e0` (heap address)
- `data=0x8c67b3a2000` (GPU address)

### 2. Usage Path (Corruption Occurs Here)

**In `build_full_attn_block()` (line 545-547):**
```cpp
fprintf(stderr, "[TriAttention] build_full_attn_block tria_k_pre_rope=%p data=%p ne=[%zu,%zu,%zu]\n",
        (void*)tria_k_pre_rope, tria_k_pre_rope->data,
        tria_k_pre_rope->ne[0], tria_k_pre_rope->ne[1], tria_k_pre_rope->ne[2]);
```

**Log output:**
```
[TriAttention] build_full_attn_block tria_k_pre_rope=0x7ffd2dd32f30 data=0x10000000000
```

### 3. Root Cause Analysis

**The pointer itself changes:**
- Allocation: `tria_k_pre_rope=0x59cac61925e0` (heap)
- Usage: `tria_k_pre_rope=0x7ffd2dd32f30` (stack)

This indicates that **the pointer value being passed to `build_full_attn_block()` is not the same as the allocated pointer**.

**Possible causes:**
1. **Compiler optimization bug**: The compiler may be incorrectly optimizing the parameter passing on HIP
2. **Stack corruption**: Something on the stack is overwriting the parameter value
3. **ABI issue**: The calling convention may not be correctly handling the parameter on this platform

**NOT the cause:**
- The `ggml_backend_alloc_ctx_tensors()` function correctly sets `tensor->data` (as shown by the valid GPU address `0x8c67b3a2000`)
- The tensor allocation itself is correct
- The `cache.tria_k_pre_rope` member is correctly stored in the `TargetCache` struct

### 4. The `data` Field Corruption

The `data` field also changes:
- Allocation: `data=0x8c67b3a2000` (valid GPU address)
- Usage: `data=0x10000000000` (invalid address)

The value `0x10000000000` is suspicious:
- It's a power of 2 (1 << 36)
- It could be a corrupted offset or uninitialized memory
- It suggests that the `ggml_tensor` structure at `0x7ffd2dd32f30` is either:
  - A partially initialized copy of the original tensor
  - Uninitialized stack memory that happens to have this value
  - The result of a memory corruption bug

### 5. Why the Current Guard Works

The guard at line 544:
```cpp
if (tria_k_pre_rope && tria_k_pre_rope->data) {
```

This would NOT prevent the crash if `tria_k_pre_rope->data == 0x10000000000` because that's a non-null value. The guard would pass, and then `ggml_view_3d()` would crash when trying to access the invalid memory.

The `nullptr` workaround at lines 1017 and 1133 bypasses this entirely by not even attempting the capture.

## Proposed Solutions

### Solution 1: Pass by Reference (May Fix the Issue)

Instead of passing `ggml_tensor * tria_k_pre_rope` by value, pass `ggml_tensor * & tria_k_pre_rope` (reference) or `ggml_tensor ** tria_k_pre_rope` (pointer to pointer).

**Change:**
```cpp
static ggml_tensor * build_full_attn_block(
    ...
    ggml_tensor * & tria_k_pre_rope  // Pass by reference
);
```

This ensures that the function receives the actual pointer from `cache.tria_k_pre_rope`, not a copy.

### Solution 2: Access Directly from Cache (Most Reliable)

Instead of passing `tria_k_pre_rope` as a parameter, access it directly from the `cache` parameter:

```cpp
// In build_full_attn_block:
// Remove the tria_k_pre_rope parameter
// Access via: cache->tria_k_pre_rope

// In build_single_layer and build_qwen35_graph:
// Don't pass tria_k_pre_rope at all
```

This requires refactoring `build_full_attn_block()` to have access to the full `TargetCache` struct.

### Solution 3: Use a Global or Thread-Local Variable (Workaround)

Store the pointer in a known location and access it from there:

```cpp
static thread_local ggml_tensor * g_tria_k_pre_rope = nullptr;
```

This is fragile but may work as a temporary workaround.

### Solution 4: Investigate Compiler Options

Try different compiler optimization levels or flags:
- `-O0` (no optimization)
- `-O2` (standard optimization)
- `-fno-omit-frame-pointer`
- `-fno-inline`

This may help identify if it's a compiler optimization bug.

## Verification Steps

1. **Add logging at multiple points** to trace where the pointer changes
2. **Use GDB to watch the pointer** through the call chain
3. **Check assembly code** for the function call to see how the parameter is passed
4. **Test with different compiler versions** to rule out compiler bugs
5. **Test on different platforms** (CUDA vs HIP) to isolate the issue

## Recommendations

1. **Short-term:** Keep the `nullptr` workaround in place
2. **Medium-term:** Try Solution 2 (access directly from cache)
3. **Long-term:** Report the bug to the compiler or GGML backend maintainers

## Additional Investigation Needed

- Disassemble the `build_full_attn_block()` call to see how the parameter is passed
- Check if there's a buffer overflow or stack corruption in the calling function
- Verify that the `TargetCache` struct is correctly aligned and has no padding issues
- Test with ASAN (Address Sanitizer) to detect memory corruption
