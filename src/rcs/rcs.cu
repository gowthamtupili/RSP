#include "rcs.cuh"
#include "../config/config.hpp"
#include <cmath>

namespace RCSEstimation {

__global__ void estimate_rcs_kernel(
    TargetProcessing::CUDATarget* targets,
    int numTargets,
    double transmittedPower,
    double transmitterGain,
    double receiverGain,
    double wavelength,
    double PI
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTargets) return;

    TargetProcessing::CUDATarget& target = targets[tid];
    double receivedPower = target.strength;
    double range = target.range;

    if (range <= 0.0) {
        target.rcs = 0.0;
        return;
    }

    double factor = pow(4.0 * PI, 3.0) * pow(range, 4.0);
    double denominator = transmittedPower * transmitterGain * receiverGain * pow(wavelength, 2.0);
    target.rcs = (receivedPower * factor) / denominator;
}

void launch_rcs_estimation(
    TargetProcessing::CUDATarget* d_targets,
    int numTargets,
    double transmittedPower,
    double transmitterGain,
    double receiverGain
) {
    double wavelength = RadarConfig::WAVELENGTH;
    double PI = RadarConfig::PI;

    int threadsPerBlock = 256;
    int blocks = (numTargets + threadsPerBlock - 1) / threadsPerBlock;

    estimate_rcs_kernel<<<blocks, threadsPerBlock>>>(
        d_targets,
        numTargets,
        transmittedPower,
        transmitterGain,
        receiverGain,
        wavelength,
        PI
    );

    cudaDeviceSynchronize(); // Optional, for timing/debugging
}

} // namespace RCSEstimation
