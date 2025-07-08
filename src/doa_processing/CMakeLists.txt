set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 11)

add_library(doa STATIC doa_processing.cu)
target_include_directories(doa PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 
target_include_directories(doa PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../data_types) 

# Maybe the below line is not required
# target_include_directories(peakdetection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../config)   