#include "ghost_removal.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace GhostRemoval {

constexpr double RELATIVE_SPEED_THRESHOLD = 5.0;

__global__ void ghost_removal_kernel(
    const TargetProcessing::CUDATarget* input,
    TargetProcessing::CUDATarget* output,
    int* filteredCount,
    int numTargets,
    double egoSpeed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTargets) return;

    const auto& tgt = input[idx];
    double delta = fabs(tgt.relativeSpeed - egoSpeed);

    if (delta <= RELATIVE_SPEED_THRESHOLD) {
        // valid target
        int outIdx = atomicAdd(filteredCount, 1);
        output[outIdx] = tgt;
    }
}

void launch_ghost_removal(
    const TargetProcessing::CUDATarget* d_inputTargets,
    TargetProcessing::CUDATarget* d_outputTargets,
    int numTargets,
    double egoSpeed,
    int* d_numFilteredTargets)
{
    int threads = 256;
    int blocks = (numTargets + threads - 1) / threads;

    cudaMemset(d_numFilteredTargets, 0, sizeof(int));

    ghost_removal_kernel<<<blocks, threads>>>(
        d_inputTargets,
        d_outputTargets,
        d_numFilteredTargets,
        numTargets,
        egoSpeed
    );

    cudaDeviceSynchronize(); // optional error check
}

} // namespace GhostRemovalGPU
