#include "target_processing.cuh"
#include "../config/config.hpp"
#include <cmath>
#include <iostream>

namespace TargetProcessing {

__device__ double device_magnitude_squared(cuDoubleComplex val) {
    return val.x * val.x + val.y * val.y;
}

__device__ double device_magnitude(cuDoubleComplex val) {
    return sqrt(device_magnitude_squared(val));
}

__device__ double device_arg(cuDoubleComplex val) {
    return atan2(val.y, val.x);
}

__device__ double compute_time_delay(const cuDoubleComplex* snap, int numReceivers) {
    // Simple heuristic: use magnitude of first sample
    if (numReceivers <= 0) return 0.0;
    double mag = device_magnitude(snap[0]);
    return mag * 1e-9; // Example scaling factor
}

__device__ double compute_doppler_shift(const cuDoubleComplex* snap, int numReceivers) {
    if (numReceivers <= 1) return 0.0;
    double dopplerShift = 0.0;
    for (int i = 1; i < numReceivers; ++i) {
        double phaseDiff = device_arg(snap[i]) - device_arg(snap[i - 1]);
        dopplerShift += phaseDiff;
    }
    dopplerShift /= (numReceivers - 1);
    return dopplerShift;
}

__global__ void detect_targets_kernel(const cuDoubleComplex* peakSnaps,
                                      DOAResult* doaResults,
                                      CUDATarget* targets,
                                      int numPeaks,
                                      int numReceivers,
                                      double wavelength,
                                      double c,
                                      double PI) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPeaks) return;

    const cuDoubleComplex* snap = &peakSnaps[idx * numReceivers];
    const DOAResult& doa = doaResults[idx];

    double azimuth = doa.azimuth;     // degrees
    double elevation = doa.elevation; // degrees

    // Calculate range using time delay
    double timeDelay = compute_time_delay(snap, numReceivers);
    double range = (c * timeDelay) / 2.0;

    // Convert angles to radians
    double azimuthRad = azimuth * PI / 180.0;
    double elevationRad = elevation * PI / 180.0;

    // Convert to Cartesian
    double x = range * cos(elevationRad) * cos(azimuthRad);
    double y = range * cos(elevationRad) * sin(azimuthRad);
    double z = range * sin(elevationRad);

    // Compute signal strength
    double strength = 0.0;
    for (int i = 0; i < numReceivers; ++i) {
        strength += device_magnitude(snap[i]);
    }

    // Compute relative speed
    double dopplerShift = compute_doppler_shift(snap, numReceivers);
    double relativeSpeed = (dopplerShift * wavelength) / 2.0;

    targets[idx] = {x, y, z, range, azimuth, elevation, strength, 0.0, relativeSpeed};
}

std::vector<CUDATarget> launch_detect_targets(const cuDoubleComplex* d_peakSnaps,
                                              DOAResult* d_doaResults,
                                              int numPeaks,
                                              int numReceivers) {
    // std::cout << "In Step 5 kernel: " << numPeaks << std::endl;
    std::vector<CUDATarget> h_targets(numPeaks);
    CUDATarget* d_targets;
    cudaMalloc(&d_targets, numPeaks * sizeof(CUDATarget));

    double wavelength = RadarConfig::WAVELENGTH;
    double c = 3e8;
    double PI = RadarConfig::PI;

    int threadsPerBlock = 256;
    int blocks = (numPeaks + threadsPerBlock - 1) / threadsPerBlock;
    detect_targets_kernel<<<blocks, threadsPerBlock>>>(d_peakSnaps, d_doaResults, d_targets,
                                                       numPeaks, numReceivers, wavelength, c, PI);
    cudaDeviceSynchronize();

    cudaMemcpy(h_targets.data(), d_targets, numPeaks * sizeof(CUDATarget), cudaMemcpyDeviceToHost);
    cudaFree(d_targets);
    

    return h_targets;
}

} // namespace TargetProcessing
