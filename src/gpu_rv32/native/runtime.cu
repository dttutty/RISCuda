#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include "include/rv32_core.cuh"
#include "include/gpu_assembler.cuh"
#include "include/bigint.cuh"

namespace {
thread_local std::string error;
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct AssemblyResult {
    int32_t value, status, steps, length, error_line;
    uint32_t code[gpu_asm::MAX_CODE];
};
__global__ void interpret(const uint32_t* code, int words, uint8_t* ram, int size,
                          int entry, int input, int budget, bool services,
                          rv32::Result* result, uint32_t* calls) {
    rv32::Machine machine;
    rv32::initialize(machine, size, entry, input);
    *calls = 0;
    for (;;) {
        *result = rv32::resume_memory(code, words, budget, ram, size, machine, services);
        if (result->status != 6) break;
        if (machine.r[17] == 0x200)
            machine.r[10] = rv32::big_mul(ram, size, machine.r[10], machine.r[11],
                                      machine.r[12], machine.r[13], machine.r[14]);
        else if (machine.r[17] == 0x201)
            machine.r[10] = rv32::decimal_row(ram, size, machine.r[10], machine.r[11],
                                              machine.r[12], machine.r[13]);
        else { result->status = 7; break; }
        ++*calls;
    }
}
__global__ void interpret_text(const char* source, int input, int budget, AssemblyResult* output) {
    output->status = gpu_asm::assemble(source, output->code, output->length, output->error_line);
    if (!output->status) {
        auto result = rv32::execute(output->code, output->length, input, budget);
        output->value = result.value; output->status = result.status; output->steps = result.steps;
    }
}
}
extern "C" const char* rv32_error() { return error.c_str(); }
extern "C" size_t rv32_assembly_size() { return sizeof(AssemblyResult); }
extern "C" int rv32_run(const uint32_t* code, int words, uint8_t* host_ram, int size,
                         int entry, int input, int budget, int services,
                         rv32::Result* host_result, float* milliseconds, uint32_t* host_calls) {
    uint32_t* device_code = nullptr;
    uint8_t* device_ram = nullptr;
    rv32::Result* result = nullptr;
    uint32_t* calls = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    int failed = 0;
    try {
        if (!code || !host_ram || size < 1 || size > 4194304 || words < 1 || words > 262144 ||
            entry < 0 || entry % 4 || entry / 4 >= words || budget < 1)
            throw std::runtime_error("invalid RV32 image, memory or budget");
        check(cudaMalloc(&device_code, words * sizeof(uint32_t)));
        check(cudaMalloc(&device_ram, size)); check(cudaMalloc(&result, sizeof(rv32::Result)));
        check(cudaMalloc(&calls, sizeof(uint32_t)));
        check(cudaMemcpy(device_code, code, words * sizeof(uint32_t), cudaMemcpyHostToDevice));
        check(cudaMemcpy(device_ram, host_ram, size, cudaMemcpyHostToDevice));
        check(cudaEventCreate(&start)); check(cudaEventCreate(&stop)); check(cudaEventRecord(start));
        interpret<<<1, 1>>>(device_code, words, device_ram, size, entry, input, budget, services, result, calls);
        check(cudaGetLastError()); check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
        check(cudaEventElapsedTime(milliseconds, start, stop));
        check(cudaMemcpy(host_result, result, sizeof(rv32::Result), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(host_calls, calls, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(host_ram, device_ram, size, cudaMemcpyDeviceToHost));
    } catch (const std::exception& e) { error = e.what(); failed = 1; }
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    cudaFree(calls); cudaFree(result); cudaFree(device_ram); cudaFree(device_code);
    return failed;
}
extern "C" int rv32_run_assembly(const char* source, int input, int budget, void* host, float* milliseconds) {
    char* device_text = nullptr;
    AssemblyResult* result = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    int failed = 0;
    try {
        if (!source || strlen(source) >= 8192 || budget < 1) throw std::runtime_error("invalid assembly source/budget");
        check(cudaMalloc(&device_text, strlen(source) + 1)); check(cudaMalloc(&result, sizeof(AssemblyResult)));
        check(cudaMemcpy(device_text, source, strlen(source) + 1, cudaMemcpyHostToDevice));
        check(cudaMemset(result, 0, sizeof(AssemblyResult)));
        check(cudaEventCreate(&start)); check(cudaEventCreate(&stop)); check(cudaEventRecord(start));
        interpret_text<<<1, 1>>>(device_text, input, budget, result);
        check(cudaGetLastError()); check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
        check(cudaEventElapsedTime(milliseconds, start, stop));
        check(cudaMemcpy(host, result, sizeof(AssemblyResult), cudaMemcpyDeviceToHost));
    } catch (const std::exception& e) { error = e.what(); failed = 1; }
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    cudaFree(result); cudaFree(device_text);
    return failed;
}
