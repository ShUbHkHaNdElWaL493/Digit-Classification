#   Shubh Khandelwal

cmake_minimum_required(VERSION 3.10)

project(DIGIT_CLASSIFIER)

add_subdirectory(
    ${CMAKE_SOURCE_DIR}/include
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/main.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC
    DATAPACK
    LAYERS
)

target_include_directories(${PROJECT_NAME} PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
)