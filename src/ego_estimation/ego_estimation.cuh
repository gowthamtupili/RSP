#ifndef EGO_MOTION_CUH
#define EGO_MOTION_CUH

#include "../target_processing/target_types.cuh"
#include <vector>
#include <cuda_runtime.h>

namespace EgoMotion {

    // Device kernel to extract valid relative speeds
    __global__ void extract_valid_relative_speeds(
        const TargetProcessing::CUDATarget* targets,
        double* validSpeeds,
        int* validCount,
        int numTargets
    );

    // Host function to estimate ego speed using CUDA
    double estimate_ego_motion_gpu(const std::vector<TargetProcessing::CUDATarget>& targets);

}

#endif // EGO_MOTION_CUH
