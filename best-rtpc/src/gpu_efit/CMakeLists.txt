set_source_files_properties(efit_kernels.hip gpu_efit.hip PROPERTIES LANGUAGE HIP)

add_library(gpu_efit STATIC
    efit_kernels.hip
    gpu_efit.hip)

target_include_directories(gpu_efit PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gpu_efit PUBLIC hip::device common)
