#ifndef DOA_PROCESSING_CUH
#define DOA_PROCESSING_CUH

#include <cuComplex.h>
#include <vector>
#include "../data_types/datatypes.cuh"

namespace DOAProcessingGPU {

    // Host function to launch DOA computation for each peakSnap
    void compute_music_doa_gpu(const cuDoubleComplex* d_peaksnaps,
                               DOAResult* h_doaResults,
                               int num_peaks,
                               int num_receivers,
                               int num_sources);

    // Device kernel to compute covariance matrix from a single snapshot
    __global__ void compute_covariance_kernel(const cuDoubleComplex* snap,
                                              cuDoubleComplex* covariance_matrix,
                                              int num_receivers);

    // Device function to compute Hermitian of a square complex matrix
    // __device__ void hermitian(const cuDoubleComplex* in_matrix,
    //                           cuDoubleComplex* out_matrix,
    //                           int dim);

    // Device kernel for simple power method-based eigenvalue extraction (1 eigenvalue/vector)
    // __global__ void power_method_kernel(cuDoubleComplex* A,
    //                                     cuDoubleComplex* eigenvector,
    //                                     double* eigenvalue,
    //                                     int n,
    //                                     int max_iters,
    //                                     double tol);

    // Device function to compute the MUSIC pseudo-spectrum over a theta/phi grid
    // __global__ void compute_music_spectrum_kernel(const cuDoubleComplex* noise_subspace,
    //                                               int num_receivers,
    //                                               int noise_dim,
    //                                               double* spectrum_grid,
    //                                               int theta_bins,
    //                                               int phi_bins,
    //                                               double d,
    //                                               double wavelength);

    // Device function to generate steering vector for given (theta, phi)
    // __device__ void compute_steering_vector(cuDoubleComplex* steering,
    //                                         int num_receivers,
    //                                         double theta_deg,
    //                                         double phi_deg,
    //                                         double d,
    //                                         double wavelength);

} // namespace DOAProcessingGPU

#endif // DOA_PROCESSING_CUH
