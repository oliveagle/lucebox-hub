// Qwen3 daemon entry point.

#pragma once

#include <string>

namespace dflash27b {

struct Qwen3DaemonArgs {
    const char * model_path = nullptr;
    int          max_ctx    = 4096;
    int          stream_fd  = -1;
    int          chunk      = 512;
    int          gpu        = 0;
};

int run_qwen3_daemon(const Qwen3DaemonArgs & args);

}  // namespace dflash27b
