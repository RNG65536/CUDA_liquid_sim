cmake_minimum_required(VERSION 3.0)
project(CUDA_liquid_sim)

enable_language(CUDA)
include_directories($ENV{NVCUDASAMPLES_ROOT}/common/inc)

if (WIN32 AND MSVC)
    add_link_options("/DEBUG")
    add_definitions("-DNOMINMAX -Xcompiler")
    add_definitions("/Zi -Xcompiler")
    add_definitions("/bigobj -Xcompiler")
    add_definitions("/wd4819 -Xcompiler")
endif()

file(GLOB_RECURSE sources "src/*.cpp" "src/*.cu")
file(GLOB_RECURSE headers "src/*.h" "src/*.cuh")

add_executable(${PROJECT_NAME} ${sources} ${headers})

target_link_libraries(${PROJECT_NAME}
	cudart.lib
	cublas.lib
	cusparse.lib
)
