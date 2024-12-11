# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name
project(CoinDetector)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add the source directory
set(SRC_DIR src)

# Set the source files
set(SOURCES ${SRC_DIR}/coin_detector.cpp)

# Create an executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Specify output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
