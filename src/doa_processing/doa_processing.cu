
// doa_processing.cu
#include "doa_processing.cuh"
#include "../config/config.hpp"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <complex>
#include <iostream>

#define DEG2RAD(x) ((x) * 3.14159265359 / 180.0)
#define PIv 3.14159265359

namespace DOAProcessingGPU {

// Utility: cuDoubleComplex multiplication and conjugate
__device__ cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ cuDoubleComplex cconj(cuDoubleComplex a) {
    return make_cuDoubleComplex(a.x, -a.y);
}


__device__ void compute_dominant_eigenvector(const cuDoubleComplex* R, cuDoubleComplex* v_out, int max_iters = 1000, double tol = 1e-6) {
    cuDoubleComplex v[3] = {
        make_cuDoubleComplex(1, 0),
        make_cuDoubleComplex(1, 0),
        make_cuDoubleComplex(1, 0)
    };

    // Normalize initial vector
    double norm = 0.0;
    for (int i = 0; i < 3; ++i) norm += v[i].x * v[i].x + v[i].y * v[i].y;
    norm = sqrt(norm);
    for (int i = 0; i < 3; ++i) {
        v[i].x /= norm;
        v[i].y /= norm;
    }

    double lambda = 0.0;

    for (int iter = 0; iter < max_iters; ++iter) {
        // Multiply: v_next = R * v
        cuDoubleComplex v_next[3] = { {0,0}, {0,0}, {0,0} };
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                v_next[i] = cuCadd(v_next[i], cmul(R[i * 3 + j], v[j]));

        // Normalize v_next
        double norm_next = 0.0;
        for (int i = 0; i < 3; ++i)
            norm_next += v_next[i].x * v_next[i].x + v_next[i].y * v_next[i].y;
        norm_next = sqrt(norm_next);

        for (int i = 0; i < 3; ++i) {
            v_next[i].x /= norm_next;
            v_next[i].y /= norm_next;
        }

        // Rayleigh quotient: lambda_next = v*† R v
        cuDoubleComplex r_dot = make_cuDoubleComplex(0, 0);
        for (int i = 0; i < 3; ++i) {
            cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
            for (int j = 0; j < 3; ++j)
                sum = cuCadd(sum, cmul(R[i * 3 + j], v[j]));
            cuDoubleComplex conj_vi = make_cuDoubleComplex(v[i].x, -v[i].y);
            r_dot = cuCadd(r_dot, cmul(conj_vi, sum));
        }

        double lambda_next = r_dot.x;  // Since imaginary part should be near-zero

        if (fabs(lambda_next - lambda) < tol)
            break;

        lambda = lambda_next;

        // Copy v_next to v
        for (int i = 0; i < 3; ++i) {
            v[i] = v_next[i];
        }
    }

    // Return result
    for (int i = 0; i < 3; ++i)
        v_out[i] = v[i];
}


// Kernel: One thread per peak
__global__ void doa_kernel(const cuDoubleComplex* peaksnaps, DOAResult* doa_out,
                           int num_peaks, int num_receivers){
    int peak_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (peak_id >= num_peaks) return;

    const cuDoubleComplex* snap = &peaksnaps[peak_id * num_receivers];
    // __syncthreads();
    // Step 1: Covariance matrix R = snap * snap^H
    cuDoubleComplex R[9]; // 3x3
    for (int i = 0; i < num_receivers; ++i) {
        for (int j = 0; j < num_receivers; ++j) {
            R[i * num_receivers + j] = cmul(snap[i], cconj(snap[j]));

            // printf("%f:%f,, ", cuCreal(R[i * num_receivers + j]), cuCimag(R[i * num_receivers + j]));
        }
    }
    // __syncthreads();
    // if(peak_id == 1) printf("end1\n\n");
    // Step 2: extract eigenvector (
    cuDoubleComplex v[3];
    compute_dominant_eigenvector(R, v);
        // for (int i = 0; i < 3; ++i) {
        //     printf("v[i] = (%f, %f) \n", peak_id, cuCreal(v[i]), cuCimag(v[i]));
        // }
        // printf("\n");

    // Step 3: Compute MUSIC spectrum over theta/phi grid
    double wavelength = RadarConfig::WAVELENGTH;
    double d = RadarConfig::ANTENNA_SPACING;
    double max_spectrum = -1.0;
    double best_theta = 0.0, best_phi = 0.0;

    for (int t = -90; t <= 90; ++t) {
        for (int p = -90; p <= 90; ++p) {
            double theta_rad = DEG2RAD(t);
            double phi_rad = DEG2RAD(p);

            // Steering vector
            cuDoubleComplex a[3];
            for (int i = 0; i < 3; ++i) {
                double phase = 2 * PIv * d * i * (sin(theta_rad) * cos(phi_rad)) / wavelength;
                a[i] = make_cuDoubleComplex(cos(phase), sin(phase));
            }

            // Projection onto noise subspace (orthogonal to signal vec v)
            cuDoubleComplex dot = make_cuDoubleComplex(0, 0);
            for (int i = 0; i < 3; ++i) {
                dot = cuCadd(dot, cmul(cconj(v[i]), a[i]));
            }

            double proj_mag = dot.x * dot.x + dot.y * dot.y;
            if (proj_mag < 1e-12) proj_mag = 1e-12;
            double spectrum = 1.0 / proj_mag;

            if (spectrum > max_spectrum) {
                max_spectrum = spectrum;
                best_theta = t;
                best_phi = p;
            }
        }
    }

    // doa_out[peak_id] = std::make_pair(best_theta, best_phi);
    doa_out[peak_id].azimuth = best_theta;
    doa_out[peak_id].elevation = best_phi;

}

void compute_music_doa_gpu(const cuDoubleComplex* d_peaksnaps,
                           DOAResult* h_doaResults,
                           int num_peaks,
                           int num_receivers,
                           int num_sources) {
    // std::pair<double, double>* d_doa;
    // cudaMalloc(&d_doa, num_peaks * sizeof(std::pair<double, double>));
    DOAResult* d_doa;
    cudaMalloc(&d_doa, num_peaks * sizeof(DOAResult));



    int threadsPerBlock = 64;
    int blocks = (num_peaks + threadsPerBlock - 1) / threadsPerBlock;

    doa_kernel<<<blocks, threadsPerBlock>>>(d_peaksnaps, d_doa, num_peaks, num_receivers);
    cudaDeviceSynchronize();

    cudaMemcpy(h_doaResults, d_doa, num_peaks * sizeof(std::pair<double, double>), cudaMemcpyDeviceToHost);
    cudaFree(d_doa);
}

} // namespace DOAProcessingGPU
