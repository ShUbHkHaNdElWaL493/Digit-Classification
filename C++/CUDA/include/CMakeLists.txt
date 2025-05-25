#   Shubh Khandelwal

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenMP REQUIRED)

add_library(DATAPACK STATIC datapack.h datapack.cpp)

add_library(NEURON STATIC neuron.h neuron.cpp)

add_library(LAYERS STATIC layers.h layers.cpp)
target_link_libraries(LAYERS PUBLIC NEURON OpenMP::OpenMP_CXX)