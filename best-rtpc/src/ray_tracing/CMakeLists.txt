set_source_files_properties(rt_kernels.hip ray_tracing.hip PROPERTIES LANGUAGE HIP)

add_library(ray_tracing STATIC
    rt_kernels.hip
    ray_tracing.hip)

target_include_directories(ray_tracing PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(ray_tracing PUBLIC hip::device common)
