// dflash27b — standalone CUDA library for DFlash speculative decoding of
// Qwen3.5-27B with the z-lab/Qwen3.5-27B-DFlash draft model on a single RTX 3090.
//
// Model constants (hardcoded for this pair) + the last-error helper.
// The real driver is test/test_dflash.cpp. A clean public API with chat /
// streaming / KV persistence is a planned contributor task (see CONTRIBUTING.md).

#ifndef DFLASH27B_H
#define DFLASH27B_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Model config ─────────────────────────────────────────────────

#define DFLASH27B_TARGET_HIDDEN        5120
#define DFLASH27B_TARGET_LAYERS        64
#define DFLASH27B_TARGET_N_HEADS       32
#define DFLASH27B_TARGET_N_KV_HEADS    8
#define DFLASH27B_TARGET_HEAD_DIM      128
#define DFLASH27B_TARGET_INTERMEDIATE  17408
#define DFLASH27B_TARGET_VOCAB         248320
#define DFLASH27B_ROPE_THETA           10000000.0f
#define DFLASH27B_RMS_EPS              1e-6f

#define DFLASH27B_DRAFT_LAYERS         5
#define DFLASH27B_DRAFT_BLOCK_SIZE     16
#define DFLASH27B_DRAFT_N_TARGET_LAYERS 5  // fc projects 5*hidden -> hidden
#define DFLASH27B_DRAFT_MASK_TOKEN_ID  248070

// target_layer_ids = {1, 16, 31, 46, 61}  (0-indexed into target layers)
// We capture the OUTPUT of each, which is HF hidden_states[lid + 1].

// ─── Diagnostics ──────────────────────────────────────────────────

// Most recent error from any loader / graph builder. Thread-safe.
const char * dflash27b_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // DFLASH27B_H
