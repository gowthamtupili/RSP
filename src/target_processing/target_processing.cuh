#ifndef TARGET_PROCESSING_CUH
#define TARGET_PROCESSING_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "../data_types/datatypes.cuh"
#include <vector>
#include "target_types.cuh"

namespace TargetProcessing {
    
    // Define a structure to represent a detected target (for use on both host and device)
    // struct CUDATarget {
    // double x, y, z;          // Cartesian coordinates
    // double range;            // Range in meters
    // double azimuth;          // Azimuth angle in degrees
    // double elevation;        // Elevation angle in degrees
    // double strength;         // Signal strength
    // double rcs;              // Radar cross section (can be filled later)
    // double relativeSpeed;    // Relative speed in m/s (from Doppler)
    // }; 

// Kernel to detect targets from peak snapshots and DOA results
__global__ void detect_targets_kernel(const cuDoubleComplex* peakSnaps,
    DOAResult* doaResults,
    CUDATarget* targets,
    int numPeaks,
    int numReceivers,
    double wavelength,
    double c,
    double PI);
    
    
    // Host launcher for target detection
    std::vector<CUDATarget> launch_detect_targets(
    const cuDoubleComplex* d_peakSnaps,
    DOAResult* d_doaResults,
    int numPeaks,
    int numReceivers
    );
    
    #endif // TARGET_PROCESSING_CUH
                                        
}