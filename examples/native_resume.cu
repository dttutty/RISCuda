// Two custom ECALLs serviced inside a single CUDA kernel, without host callbacks.
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include "rv32_core.cuh"

struct Outcome { int value; unsigned status, steps, services; };
__global__ void guest(Outcome* output, int budget) {
    const uint32_t code[] = {
        0x01500513,  // li a0,21
        0x02a00893,  // li a7,42 -- custom service: double a0
        0x00000073,  // ecall
        0x00150513,  // addi a0,a0,1
        0x00000073,  // ecall
        0x00000893,  // li a7,0 -- halt
        0x00000073,
    };
    uint8_t ram[1024] = {};
    rv32::Machine state;
    rv32::initialize(state, sizeof(ram));
    rv32::Result result;
    unsigned services = 0;
    do {
        result = rv32::resume_memory(code, 7, budget, ram, sizeof(ram), state, true);
        if (result.status == 6) {
            state.r[10] *= 2;
            ++services;
        }
    } while (result.status == 6);
    *output = {result.value, result.status, result.steps, services};
}
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
int main(int argc, char** argv) {
    try {
        int budget = argc > 1 ? std::stoi(argv[1]) : 100;
        if (budget < 1) throw std::runtime_error("budget must be positive");
        Outcome *device = nullptr, result{};
        check(cudaMalloc(&device, sizeof(Outcome)));
        guest<<<1, 1>>>(device, budget);
        check(cudaGetLastError());
        check(cudaMemcpy(&result, device, sizeof(result), cudaMemcpyDeviceToHost));
        check(cudaFree(device));
        std::cout << "{\"value\":" << result.value << ",\"status\":" << result.status
                  << ",\"steps\":" << result.steps << ",\"services\":" << result.services << "}\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
