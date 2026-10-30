#pragma once
#include <cstdint>

namespace resident {
constexpr int TOKENS=8192, QUEUE=512, CALLS=64, ROWS=80;
struct Tool {
    int32_t status, error, steps, bigint_calls, gather_error, input_length, output_length;
    char expression[256], output[256];
    uint64_t started, elapsed_ns;
};
struct State {
    int32_t position, forwards, generated, injected, total, calls, stop, pending;
    int32_t armed, cap, queue_index, queue_length;
    int32_t tokens[TOKENS], queue[QUEUE];
    uint8_t forced[TOKENS], rows[ROWS+1][16];
    Tool tools[CALLS];
};
// Buffer and graph pointers are device resources owned by the embedding app.
// code/initial point at HOST bytes during create; private device copies are made.
struct Options {
    uint64_t prefill_graph, decode_graph;
    void *input, *next, *position, *mask, *bytes, *offsets, *printable;
    const void *code, *initial;
    int32_t vocab, cache_length, prompt_length, max_generated, max_total, max_calls;
    int32_t eos, eos2, tco, tcc, words, ram_size, entry, source, output, error, budget;
};
} // namespace resident
