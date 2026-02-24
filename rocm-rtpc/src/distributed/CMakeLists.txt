add_library(distributed STATIC
    rfm_transport.cpp)

target_include_directories(distributed PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(distributed PUBLIC common)
