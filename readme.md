# Coin Detection Application

This project is a C++ application for detecting and counting coins in images using OpenCV. The application processes grayscale and color images, highlights detected coins with their center and boundary, and provides detailed results about the coin positions and sizes.

## Features
- Detect coins in images and mark their centers and boundaries.
- Save processed images with results in the `results` folder.
- Print the total number of coins and their center coordinates (x, y) and radius.
- Evaluate the accuracy of the detection using a dataset with manual annotations.


## Requirements
- **C++17 or higher**
- **OpenCV 4.0 or higher**
- **CMake 3.15 or higher**

## Installation
1. Open a terminal or command prompt.
2. Clone the repository git@github.com:LeinMS/coins-detection.git
3. Create and navigate to the build directory
```bash
mkdir build
cd build
```
4. Run CMake to generate the build files and build the project
```bash
cmake ..
cmake --build .
```
5. Execute the program
```bash
cd .\Debug\
./CoinDetector.exe D:\VSCode_projects\2d_image_proc\data\images\
```
It creates the the 'result' folder and saves the processed images and data there.