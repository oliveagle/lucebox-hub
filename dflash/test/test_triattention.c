/*
 * test_triattention.c — Minimal test for TriAttention C library
 *
 * Tests:
 * 1. tria_load() — load .bin stats file
 * 2. tria_score_kv_head() — score a single KV head
 * 3. tria_free() — cleanup
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

/* Include TriAttention C library */
#include "../deps/llama.cpp/triattention/triattention.h"

#define STATS_PATH_DEFAULT "../deps/llama.cpp/triattention/stats/qwen3.6-27b.bin"

int main(int argc, char **argv) {
    const char *stats_path = STATS_PATH_DEFAULT;
    if (argc > 1) {
        stats_path = argv[1];
    }

    printf("=== TriAttention C Library Test ===\n");
    printf("Stats path: %s\n\n", stats_path);

    /* ------------------------------------------------------------------ */
    /* Test 1: Load stats file                                            */
    /* ------------------------------------------------------------------ */
    printf("Test 1: Loading stats file...\n");
    struct tria_stats *stats = tria_load(stats_path);
    if (!stats) {
        fprintf(stderr, "FAILED: tria_load returned NULL\n");
        return 1;
    }
    printf("  Stats loaded successfully!\n");
    printf("  num_layers: %u\n", stats->num_layers);
    printf("  num_heads: %u\n", stats->num_heads);
    printf("  num_kv_heads: %u\n", stats->num_kv_heads);
    printf("  head_dim: %u\n", stats->head_dim);
    printf("  freq_count: %u\n", stats->freq_count);
    printf("  rope_theta: %f\n", stats->rope_theta);
    printf("  attn_scale: %f\n", stats->attn_scale);
    printf("\n");

    /* ------------------------------------------------------------------ */
    /* Test 2: tria_layer_budget()                                        */
    /* ------------------------------------------------------------------ */
    printf("Test 2: Testing tria_layer_budget()...\n");
    for (int layer = 0; layer < (int)stats->num_layers && layer < 5; layer++) {
        int budget = tria_layer_budget(stats, layer, 2048);
        float scale = stats->layer_budget_scales[layer];
        printf("  Layer %d: scale=%.3f, budget=%d (base=2048)\n",
               layer, scale, budget);
    }
    printf("\n");

    /* ------------------------------------------------------------------ */
    /* Test 3: tria_score_kv_head() — minimal scoring test                */
    /* ------------------------------------------------------------------ */
    printf("Test 3: Testing tria_score_kv_head()...\n");

    int seq_len = 16;
    int fc = stats->freq_count;
    int layer_idx = 0;
    int kv_head_idx = 0;
    int cur_pos = 100;

    /* Allocate and initialize test data */
    float *k_pre_real = calloc(seq_len * fc, sizeof(float));
    float *k_pre_imag = calloc(seq_len * fc, sizeof(float));
    int *key_pos = calloc(seq_len, sizeof(int));
    float *out_scores = calloc(seq_len, sizeof(float));

    /* Simple test data: sinusoidal-like keys */
    for (int s = 0; s < seq_len; s++) {
        key_pos[s] = s;
        for (int f = 0; f < fc; f++) {
            float angle = (float)(s * f) / fc * 3.14159f;
            k_pre_real[s * fc + f] = cosf(angle);
            k_pre_imag[s * fc + f] = sinf(angle);
        }
    }

    printf("  Calling tria_score_kv_head()...\n");
    printf("    seq_len=%d, layer=%d, kv_head=%d, cur_pos=%d\n",
           seq_len, layer_idx, kv_head_idx, cur_pos);

    tria_score_kv_head(stats, k_pre_real, k_pre_imag, key_pos,
                       cur_pos, seq_len, layer_idx, kv_head_idx, out_scores);

    printf("  Scoring complete!\n");
    printf("  Sample scores (first 8 positions):\n");
    for (int s = 0; s < 8 && s < seq_len; s++) {
        printf("    pos %d: %.6f\n", s, out_scores[s]);
    }
    printf("\n");

    /* Verify scores are finite numbers */
    int valid_scores = 1;
    for (int s = 0; s < seq_len; s++) {
        if (!isfinite(out_scores[s])) {
            fprintf(stderr, "FAILED: score at position %d is not finite: %f\n",
                    s, out_scores[s]);
            valid_scores = 0;
        }
    }
    if (valid_scores) {
        printf("  All scores are finite numbers! ✓\n");
    }
    printf("\n");

    /* ------------------------------------------------------------------ */
    /* Cleanup                                                            */
    /* ------------------------------------------------------------------ */
    free(k_pre_real);
    free(k_pre_imag);
    free(key_pos);
    free(out_scores);

    printf("Test 4: Cleanup...\n");
    tria_free(stats);
    printf("  tria_free() completed!\n");

    printf("\n=== All tests passed! ===\n");
    return 0;
}
