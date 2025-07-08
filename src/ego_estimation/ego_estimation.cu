#include "ego_estimation.cuh"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <numeric>
#include <cmath>

namespace EgoMotion {

    __global__ void extract_valid_relative_speeds(
        const TargetProcessing::CUDATarget* targets,
        double* validSpeeds,
        int* validCount,
        int numTargets
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numTargets) return;

        double speed = targets[idx].relativeSpeed;
        if (fabs(speed) > 0.1) {
            int insertIdx = atomicAdd(validCount, 1);
            validSpeeds[insertIdx] = speed;
        }
    }

    double estimate_ego_motion_gpu(const std::vector<TargetProcessing::CUDATarget>& h_targets) {
        if (h_targets.empty()) return 0.0;

        int numTargets = h_targets.size();
        TargetProcessing::CUDATarget* d_targets;
        double* d_validSpeeds;
        int* d_validCount;
        int maxValid = numTargets;

        cudaMalloc(&d_targets, sizeof(TargetProcessing::CUDATarget) * numTargets);
        cudaMemcpy(d_targets, h_targets.data(), sizeof(TargetProcessing::CUDATarget) * numTargets, cudaMemcpyHostToDevice);
        cudaMalloc(&d_validSpeeds, sizeof(double) * maxValid);
        cudaMalloc(&d_validCount, sizeof(int));
        cudaMemset(d_validCount, 0, sizeof(int));

        int threadsPerBlock = 256;
        int blocks = (numTargets + threadsPerBlock - 1) / threadsPerBlock;
        extract_valid_relative_speeds<<<blocks, threadsPerBlock>>>(
            d_targets, d_validSpeeds, d_validCount, numTargets);
        cudaDeviceSynchronize();

        // Copy back valid count
        int h_validCount = 0;
        cudaMemcpy(&h_validCount, d_validCount, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_validCount == 0) {
            cudaFree(d_targets); cudaFree(d_validSpeeds); cudaFree(d_validCount);
            return 0.0;
        }

        std::vector<double> h_validSpeeds(h_validCount);
        cudaMemcpy(h_validSpeeds.data(), d_validSpeeds, sizeof(double) * h_validCount, cudaMemcpyDeviceToHost);

        double sum = std::accumulate(h_validSpeeds.begin(), h_validSpeeds.end(), 0.0);
        double avg = sum / h_validCount;

        cudaFree(d_targets); cudaFree(d_validSpeeds); cudaFree(d_validCount);
        return avg;
    }

}
