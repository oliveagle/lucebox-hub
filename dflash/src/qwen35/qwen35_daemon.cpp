// Qwen35 daemon entry point.
//
// Thin wrapper: constructs a Qwen35Backend and hands off to the generic
// daemon loop (daemon_loop.cpp). All model-specific logic lives in
// qwen35_backend.cpp; protocol plumbing lives in daemon_loop.cpp.

#include "qwen35_daemon.h"
#include "qwen35_backend.h"
#include "common/daemon_loop.h"

#if defined(DFLASH27B_TRIATTENTION_ENABLED)
#include "triattention_runner.h"
#endif

#include <cstdio>

namespace dflash27b {

int run_qwen35_daemon(const Qwen35DaemonArgs & args) {
    std::fprintf(stderr, "[DEBUG] run_qwen35_daemon(): entering\n"); std::fflush(stderr);
#if defined(DFLASH27B_TRIATTENTION_ENABLED)
    // Initialize TriAttention KV compression from environment variables.
    // This loads the stats file and configures compression parameters.
    std::fprintf(stderr, "[DEBUG] run_qwen35_daemon(): calling init_triattention_from_env()\n"); std::fflush(stderr);
    init_triattention_from_env();
    if (g_tria_state.enabled) {
        std::fprintf(stderr, "[TriAttention] Enabled with stats loaded\n");
    } else {
        std::fprintf(stderr, "[TriAttention] Disabled (stats not found or load failed)\n");
    }
#endif

    std::fprintf(stderr, "[DEBUG] run_qwen35_daemon(): building config\n"); std::fflush(stderr);
    Qwen35Config cfg;
    cfg.target_path        = args.target_path;
    cfg.draft_path         = args.draft_path;
    cfg.device             = args.device;
    cfg.draft_gpu          = args.draft_gpu;
    cfg.stream_fd          = args.stream_fd;
    cfg.fa_window          = args.fa_window;
    cfg.kq_stride_pad      = args.kq_stride_pad;
    cfg.draft_swa_window   = args.draft_swa_window;
    cfg.draft_ctx_max      = args.draft_ctx_max;
    cfg.fast_rollback      = args.fast_rollback;
    cfg.seq_verify         = args.seq_verify;
    cfg.ddtree_mode        = args.ddtree_mode;
    cfg.ddtree_budget      = args.ddtree_budget;
    cfg.ddtree_temp        = args.ddtree_temp;
    cfg.ddtree_chain_seed  = args.ddtree_chain_seed;
    cfg.use_feature_mirror = args.use_feature_mirror;

    std::fprintf(stderr, "[DEBUG] run_qwen35_daemon(): creating Qwen35Backend\n"); std::fflush(stderr);
    Qwen35Backend backend(cfg);
    std::fprintf(stderr, "[DEBUG] run_qwen35_daemon(): calling backend.init()\n"); std::fflush(stderr);
    if (!backend.init()) {
        std::fprintf(stderr, "[DEBUG] run_qwen35_daemon(): backend.init() FAILED\n"); std::fflush(stderr);
        return 1;
    }
    std::fprintf(stderr, "[DEBUG] run_qwen35_daemon(): backend.init() OK\n"); std::fflush(stderr);

    DaemonLoopArgs dargs;
    dargs.stream_fd = args.stream_fd;
    dargs.chunk     = args.chunk;
    dargs.max_ctx   = args.device.max_ctx;

    std::fprintf(stderr, "[DEBUG] run_qwen35_daemon(): entering daemon loop\n"); std::fflush(stderr);
    return run_daemon(backend, dargs);
}

}  // namespace dflash27b
