add_library(common STATIC
    plasma_profiles.cpp)

target_include_directories(common PUBLIC ${CMAKE_SOURCE_DIR}/include)
