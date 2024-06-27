cmake_minimum_required(VERSION 3.17)

project(atom_profiles LANGUAGES CUDA)
set(CMAKE_CUDA_ARCHITECTURES "70;80")

add_executable(atom_profiles
	kernel.cu
)
target_compile_options(atom_profiles PUBLIC
	-march=native
	-Wfloat-conversion
	-Wall -Wextra
)
