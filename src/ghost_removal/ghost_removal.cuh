#ifndef GHOST_REMOVAL_CUH
#define GHOST_REMOVAL_CUH

#include "../target_processing/target_types.cuh"

namespace GhostRemoval {
    // Launch kernel to remove ghost targets on GPU
    void launch_ghost_removal(
        const TargetProcessing::CUDATarget* d_inputTargets,
        TargetProcessing::CUDATarget* d_outputTargets,
        int numTargets,
        double egoSpeed,
        int* d_numFilteredTargets // output count
    );
}

#endif // GHOST_REMOVAL_CUH
