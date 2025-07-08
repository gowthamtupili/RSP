#include <iostream>
#include <chrono> // Include for timing functions
#include "config/config.hpp"
#include "data_types/datatypes.cuh"
#include "preprocessing/fft_processing.cuh"
#include "peak_detection/peak_detection.cuh"
#include "mimo_synthesis/mimo_synthesis.cuh"
#include "doa_processing/doa_processing.cuh"
#include "target_processing/target_processing.cuh" 
#include "rcs/rcs.cuh"
#include "ego_estimation/ego_estimation.cuh"
#include "ghost_removal/ghost_removal.cuh"


#include <algorithm>

int main() 
{
    // Load radar configuration

    RadarConfig::Config rconfig = RadarConfig::load_config();
    RadarData::Frame frame(rconfig.num_receivers, rconfig.num_chirps, rconfig.num_samples);
    std::cout << "Radar Configuration Loaded:" << std::endl;
    RadarData::peakInfo peakinfo(rconfig.num_receivers, rconfig.num_chirps, rconfig.num_samples);

     // Number of frames to process
    constexpr int NUM_FRAMES = 1;
    for (int frameIndex = 0; frameIndex < NUM_FRAMES; ++frameIndex) {
        std::cout << "Processing frame " << frameIndex + 1 << " of " << NUM_FRAMES << std::endl;

        // Initialize frame by reading data for the current frame
        RadarData::initialize_frame(
            frame,
            rconfig.num_receivers,
            rconfig.num_chirps,
            rconfig.num_samples,
            frameIndex
        );

        //std::cout << "Data Initialized" << std::endl;
        // Calculate frame size in bytes
        size_t frame_size = RadarData::frame_size_bytes(frame);
        //std::cout << "Frame size in bytes: " << frame_size << std::endl;
        frame.copy_frame_to_device();
      
       
        //*********************STEP 1 FFT PROCESSING *******************
        auto start = std::chrono::high_resolution_clock::now();
        fftProcessing::fftProcessPipeline(frame);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time taken for fftProcessPipeline: " << elapsed.count() << " seconds" << std::endl;
        frame.copy_frame_to_host();
    
    //*********************STEP 2 PEAK DETECTION  *******************

       start = std::chrono::high_resolution_clock::now();
       PeakDetection::cfar_peak_detection(frame, peakinfo);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        
        std::cout << "Time taken for peakDetection: " << elapsed.count() << " seconds" << std::endl;
        peakinfo.copy_peakInfo_to_host();
        std::cout << "Number of peaks detected: " << peakinfo.num_peaks << std::endl;
        // Output detected peaks
        /*for (int i = 0; i < peakinfo.num_peaks; ++i) {
            const RadarData::Peak& peak = peakinfo.peakList[i];
            std::cout << "Peak " << i + 1 << ": Receiver " << peak.receiver
                      << ", Chirp " << peak.chirp
                      << ", Sample " << peak.sample
                      << ", Value " << peak.value << std::endl;
        }*/
    
        //*********************STEP 3 MIMO SYNTHESIS PEAK SNAP DETECTION  *******************
        start = std::chrono::high_resolution_clock::now();
        MIMOSynthesis::synthesize_peaks(frame, peakinfo);
        peakinfo.copyPeakSnapsToHost();
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Time taken for MIMO synthesis: " << elapsed.count() << " seconds" << std::endl;
        // Output synthesized peak snaps
        // for (int i = 0; i < peakinfo.num_peaks; ++i) {
            // std::cout << "Peak Snap " << i + 1 << ": ";
            // for (int r = 0; r < rconfig.num_receivers; ++r) {
                // std::cout << peakinfo.peaksnaps[i * rconfig.num_receivers + r] << " ";
            // }
            //std::cout << std::endl;
        // }
    
     
        //*********************STEP 4 DOA PROCESSING  *******************
        // std::vector<std::pair<double, double>> doaResults(peakinfo.num_peaks);
        std::vector<DOAResult> doaResults(peakinfo.num_peaks);


        start = std::chrono::high_resolution_clock::now();

        // Flatten peakinfo.peaksnaps to cuDoubleComplex*
        std::vector<cuDoubleComplex> h_flat_peaksnaps(peakinfo.num_peaks * rconfig.num_receivers);
        for (int i = 0; i < peakinfo.num_peaks; ++i) {
            for (int j = 0; j < rconfig.num_receivers; ++j) {
                std::complex<double>& val = peakinfo.peaksnaps[i * rconfig.num_receivers + j];
                h_flat_peaksnaps[i * rconfig.num_receivers + j] = make_cuDoubleComplex(val.real(), val.imag());
            }
        }

        cuDoubleComplex* d_peaksnaps = nullptr;
        cudaMalloc(&d_peaksnaps, h_flat_peaksnaps.size() * sizeof(cuDoubleComplex));
        cudaMemcpy(d_peaksnaps, h_flat_peaksnaps.data(), h_flat_peaksnaps.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        DOAProcessingGPU::compute_music_doa_gpu(d_peaksnaps, doaResults.data(), peakinfo.num_peaks, rconfig.num_receivers, 1);
        // cudaFree(d_peaksnaps);

        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;
        std::cout << "Time taken for DOA Processing: " << elapsed.count() << " seconds" << std::endl;
        // Output DOA results for the current frame
        // std::cout << "DOA Results (Azimuth, Elevation) for frame " << frameIndex + 1 << ":" <<doaResults.size()<<std::endl;
        // for (const auto& result : doaResults) {
        //     std::cout << "(" << result.first << ", " << result.second << ")" << std::endl;
        // }
        // std::sort(doaResults.begin(), doaResults.end(),
        //   [](const DOAResult& a, const DOAResult& b) {
        //       return a.azimuth < b.azimuth;
        //   });
        //   if(frameIndex == 0) {
        //       for (const auto& result : doaResults) {
        //           std::cout << "(" << result.azimuth << ", " << result.elevation << ")" << std::endl;
        //       }
        //   }

        // std::cout << "peaksnap size = " << peakSnaps.size() << std::endl;
    
        //*********************STEP 5 TARGET DETECTION *******************
        start = std::chrono::high_resolution_clock::now();
        DOAResult* d_doaResults;
        cudaMalloc(&d_doaResults, peakinfo.num_peaks * sizeof(DOAResult));
        cudaMemcpy(d_doaResults, doaResults.data(), peakinfo.num_peaks * sizeof(DOAResult), cudaMemcpyHostToDevice);
        std::vector<TargetProcessing::CUDATarget> h_targets = TargetProcessing::launch_detect_targets(
            d_peaksnaps,
            d_doaResults,
            peakinfo.num_peaks,
            rconfig.num_receivers
        );
        // for (int i = 0; i < std::min(5, (int)h_targets.size()); ++i) {
        //     const auto& t = h_targets[i];
        //     std::cout << "Detected target " << i << ": x=" << t.x << " y=" << t.y
        //             << " z=" << t.z << " speed=" << t.relativeSpeed << " rcs=" << t.rcs << std::endl;
        // }
        cudaFree(d_peaksnaps);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        
        // std::cout << "Targets detected:" << std::endl;
        // for (const auto& target : h_targets) {
        //     std::cout << "Location: (" << target.x << ", " << target.y << ", " << target.z << ")"
        //     << ", Range: " << target.range
        //     << ", Azimuth: " << target.azimuth
        //     << ", Elevation: " << target.elevation
        //     << ", Strength: " << target.strength
        //     << ", Relative Speed: " << target.relativeSpeed << std::endl;
        // }
        
        std::cout << "Time taken for Target Detection: " << elapsed.count() << " seconds" << std::endl;
        // std::cout << "\n\n\nhey\n\n";
        TargetProcessing::CUDATarget* d_targets = nullptr;
    cudaMalloc(&d_targets, peakinfo.num_peaks * sizeof(TargetProcessing::CUDATarget));
    cudaMemcpy(d_targets, h_targets.data(), peakinfo.num_peaks * sizeof(TargetProcessing::CUDATarget), cudaMemcpyHostToDevice);

        
        /*********************STEP 6 RADAR CROSS SECTION *******************/
        // Example radar parameters

    double transmittedPower = 1.0; // Example: 1 Watt
    double transmitterGain = 10.0; // Example: 10 dB
    double receiverGain = 10.0;    // Example: 10 dB
    
    // ----------------------------- OLD CPU CODE (SHOULD BE REMOVED BEFORE COMMIT) (STARTING POINT)
    // Detect targets
    // TargetProcessing::TargetList targets = TargetProcessing::detect_targets(peakSnaps, doaResults);
    
    // // Estimate RCS for each target
    // RCSEstimation::estimate_rcs(targets, transmittedPower, transmitterGain, receiverGain);
    // for (const auto& target : targets) {
        //std::cout << "Target RCS: " << target.rcs << " m^2" << std::endl;
    // }
// ------------------------------------------ OLD CPU CODE (TO BE REMOVED) (ENDING POINT)
    // Output results
    start = std::chrono::high_resolution_clock::now();
    RCSEstimation::launch_rcs_estimation(
    d_targets,                          // CUDA targets from target detection
    peakinfo.num_peaks,                // Number of peaks (targets)
    transmittedPower,
    transmitterGain,
    receiverGain
    );
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken for RCS (RADAR CROSS SECTION): " << elapsed.count() << " seconds" << std::endl;

// Copy updated targets back to host
// std::vector<TargetProcessing::CUDATarget> h_targets(peakinfo.num_peaks);
cudaMemcpy(h_targets.data(), d_targets, peakinfo.num_peaks * sizeof(TargetProcessing::CUDATarget), cudaMemcpyDeviceToHost);

// Optionally print
// for (const auto& target : h_targets) {
//     std::cout << "Target RCS: " << target.rcs << " m^2" << std::endl;
// }

    /*********************STEP 6 EGO ESTIMATION *******************/
    // double egoSpeed = EgoMotion::estimate_ego_motion(targetList);
    start = std::chrono::high_resolution_clock::now();
    double egoSpeed = EgoMotion::estimate_ego_motion_gpu(h_targets);  // Use GPU version
    // std::cout << "Estimated Ego Vehicle Speed: " << egoSpeed << " m/s" << std::endl;
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken for Ego Estimation: " << elapsed.count() << " seconds" << std::endl;
    // *********************STEP 7 GHOST TARGET REMOVAL *******************

        TargetProcessing::CUDATarget* d_filteredTargets;
        int* d_numFilteredTargets;
        cudaMalloc(&d_filteredTargets, h_targets.size() * sizeof(TargetProcessing::CUDATarget));
        cudaMalloc(&d_numFilteredTargets, sizeof(int));
        start = std::chrono::high_resolution_clock::now();
        GhostRemoval::launch_ghost_removal(
            d_targets,
            d_filteredTargets,
            static_cast<int>(h_targets.size()),
            egoSpeed,
            d_numFilteredTargets
        );

        // Copy number of filtered targets
        int h_numFilteredTargets = 0;
        cudaMemcpy(&h_numFilteredTargets, d_numFilteredTargets, sizeof(int), cudaMemcpyDeviceToHost);

        // Copy filtered targets
        std::vector<TargetProcessing::CUDATarget> h_filteredTargets(h_numFilteredTargets);
        cudaMemcpy(h_filteredTargets.data(), d_filteredTargets,
                h_numFilteredTargets * sizeof(TargetProcessing::CUDATarget), cudaMemcpyDeviceToHost);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Time taken for Ghost Target Removal: " << elapsed.count() << " seconds" << std::endl;
        // Output filtered targets
        std::sort(h_filteredTargets.begin(), h_filteredTargets.end(),
          [](const TargetProcessing::CUDATarget& a, const TargetProcessing::CUDATarget& b) {
              return a.x < b.x;
          });

        // Output filtered targets
        std::cout << "Filtered Targets (after ghost removal, sorted by x):" << std::endl;

        // std::cout << "Filtered Targets (after ghost removal):" << std::endl;
        for (const auto& target : h_filteredTargets) {
            // std::cout << "Location: (" << target.x << ", " << target.y << ", " << target.z << ")\n";
            // std::cout << "Location: (" << target.z  << ")\n";
                    std::cout << "Location: (" << target.x << ", " << target.y << ", " << target.z << ")"
            << ", Range: " << target.range
            << ", Azimuth: " << target.azimuth
            << ", Elevation: " << target.elevation
            << ", Strength: " << target.strength
            << ", Relative Speed: " << target.relativeSpeed 
            << std::endl;
        }
        // std::cout << "Number of targets after ghost removal: " << h_filteredTargets.size() << std::endl;

        cudaFree(d_filteredTargets);
        cudaFree(d_numFilteredTargets);

        // Keep the terminal display until a key is pressed
        // std::cout << "Processing complete. Press any key to exit..." << std::endl;
        // std::cin.get();
        }   

    return 0;
}
