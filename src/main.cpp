#include "qwen/weights.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::string weights_path;
};

void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --weights PATH\n";
}

Args parse(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--weights") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --weights");
            }
            args.weights_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (args.weights_path.empty()) {
        usage(argv[0]);
        throw std::runtime_error("--weights is required");
    }
    return args;
}

}

int main(int argc, char** argv) {
    try {
        const Args args = parse(argc, argv);
        const ModelWeights weights = load(args.weights_path);
        const ModelConfig& config = weights.config;

        std::cout << "loaded weights\n"
                  << "  hidden_size=" << config.hidden_size << "\n"
                  << "  layers=" << config.num_hidden_layers << "\n"
                  << "  heads=" << config.num_attention_heads << "\n"
                  << "  kv_heads=" << config.num_key_value_heads << "\n"
                  << "  vocab=" << config.vocab_size << "\n"
                  << "  eos_token_id=" << config.eos_token_id << "\n"
                  << "  embed_shape=[" << weights.embed_tokens.shape[0]
                  << ", " << weights.embed_tokens.shape[1] << "]\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
