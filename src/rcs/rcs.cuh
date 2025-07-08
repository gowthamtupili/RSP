#ifndef RCS_ESTIMATION_CUH
#define RCS_ESTIMATION_CUH

#include "../target_processing/target_types.cuh"

namespace RCSEstimation {

    // Kernel to estimate RCS values for each target
    __global__ void estimate_rcs_kernel(
        TargetProcessing::CUDATarget* targets,
        int numTargets,
        double transmittedPower,
        double transmitterGain,
        double receiverGain,
        double wavelength,
        double PI
    );

    // Host launcher
    void launch_rcs_estimation(
        TargetProcessing::CUDATarget* d_targets,
        int numTargets,
        double transmittedPower,
        double transmitterGain,
        double receiverGain
    );

} // namespace RCSEstimation

#endif // RCS_ESTIMATION_CUH
