#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem> // For directory creation

struct CoinAnnotation {
    float x;
    float y;
    float r;
};

static float euclideanDistance(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_directory>" << std::endl;
        std::cout << "Example: " << argv[0] << " ../data/images/\n";
        return -1;
    }

    // Ensure results directory exists
    std::string resultsDir = "../../results/";
    std::filesystem::create_directory(resultsDir);

    // File for metrics
    std::string metricsFile = resultsDir + "metrics.txt";
    std::ofstream metricsOut(metricsFile, std::ios::out | std::ios::trunc); // Overwrite file
    if (!metricsOut.is_open()) {
        std::cerr << "Error: Unable to create metrics file at " << metricsFile << "\n";
        return -1;
    }

    // Suppress OpenCV debug messages
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // Directory containing images
    std::string imageDir = argv[1];
    if (!std::filesystem::is_directory(imageDir)) {
        std::cerr << "Error: The provided path is not a directory: " << imageDir << "\n";
        return -1;
    }

    // Counters for overall statistics
    int totalTP = 0;
    int totalFP = 0;
    int totalFN = 0;
    int totalGT = 0; // total ground-truth coins

    // Iterate over all image files in the directory
    for (const auto& entry : std::filesystem::directory_iterator(imageDir)) {
        if (!entry.is_regular_file()) continue;

        std::string filename = entry.path().string();
        std::string extension = entry.path().extension().string();
        
        // Check if the file has a valid image extension
        if (extension != ".jpg" && extension != ".png" && extension != ".jpeg" && extension != ".bmp") {
            continue;
        }

        std::cout << "Processing image: " << filename << "\n";
        cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Cannot open image: " << filename << std::endl;
            continue;
        }

        // Preprocessing
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::medianBlur(gray, gray, 5);

        // Detect circles using Hough Transform
        std::vector<cv::Vec3f> circles;
        int param1 = 100; // Canny edge detection high threshold
        int param2 = 30;  // Accumulator threshold for circle detection
        int minRadius = 30;
        int maxRadius = 50;
        double rad = gray.rows / 16;
        cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, rad, param1, param2, minRadius, maxRadius);

        // Load ground truth if exists
        std::vector<CoinAnnotation> groundTruth;
        std::string annotationFile = filename.substr(0, filename.find_last_of('.')) + ".txt";
        std::ifstream fin(annotationFile);
        
        if (fin.is_open()) {
            float x, y, r;
            while (fin >> x >> y >> r) {
                groundTruth.push_back({x, y, r});
            }
        }

        // Draw detected circles
        for (size_t j = 0; j < circles.size(); j++) {
            cv::Point center(cvRound(circles[j][0]), cvRound(circles[j][1]));
            int radius = cvRound(circles[j][2]);
            cv::circle(img, center, 3, cv::Scalar(0, 255, 0), -1);
            cv::circle(img, center, radius, cv::Scalar(0, 0, 255), 2);
        }

        // Matching with ground truth if available
        int TP = 0, FP = 0, FN = 0;

        if (!groundTruth.empty()) {
            totalGT += (int)groundTruth.size();
            std::vector<bool> matchedGT(groundTruth.size(), false);
            std::vector<bool> matchedDet(circles.size(), false);

            // Match detected circles to ground truth
            for (size_t d = 0; d < circles.size(); d++) {
                float dx = circles[d][0];
                float dy = circles[d][1];
                float dr = circles[d][2];

                int bestMatch = -1;
                float bestDist = 1e9;
                for (size_t g = 0; g < groundTruth.size(); g++) {
                    float gx = groundTruth[g].x;
                    float gy = groundTruth[g].y;
                    float gr = groundTruth[g].r;
                    float dist = euclideanDistance(dx, dy, gx, gy);

                    float maxCenterDist = 0.4f * gr; // Within 40% of radius
                    float maxRadiusDiff = 0.4f * gr; // Radius within 40% difference

                    if (dist < maxCenterDist && std::fabs(dr - gr) < maxRadiusDiff) {
                        if (dist < bestDist) {
                            bestDist = dist;
                            bestMatch = (int)g;
                        }
                    }
                }

                if (bestMatch != -1 && !matchedGT[bestMatch]) {
                    matchedGT[bestMatch] = true;
                    matchedDet[d] = true;
                }
            }

            // Count TP, FP, FN
            for (bool mg : matchedGT) {
                if (mg) TP++;
            }
            for (bool md : matchedDet) {
                if (!md) FP++;
            }
            for (bool mg : matchedGT) {
                if (!mg) FN++;
            }

            totalTP += TP;
            totalFP += FP;
            totalFN += FN;

            metricsOut << "Image: " << filename << "\n";
            metricsOut << "Ground Truth: " << groundTruth.size() << "\n";
            metricsOut << "Detected Circles: " << circles.size() << "\n";
            metricsOut << "True Positives: " << TP << "\n";
            metricsOut << "False Positives: " << FP << "\n";
            metricsOut << "False Negatives: " << FN << "\n";

            metricsOut << "Ground Truth Data:\n";
            for (const auto& gt : groundTruth) {
                metricsOut << "    x=" << gt.x << ", y=" << gt.y << ", r=" << gt.r << "\n";
            }

            metricsOut << "Detected Circles:\n";
            for (const auto& circle : circles) {
                metricsOut << "    x=" << circle[0] << ", y=" << circle[1] << ", r=" << circle[2] << "\n";
            }

            if ((TP + FP) > 0) {
                float precision = float(TP) / float(TP + FP);
                float recall = float(TP) / float(TP + FN);
                float f1 = 2.0f * precision * recall / (precision + recall);

                metricsOut << "Precision: " << precision << "\n";
                metricsOut << "Recall: " << recall << "\n";
                metricsOut << "F1-score: " << f1 << "\n";
                metricsOut << "===============================================================\n";
            }
        }

        // Save output image
        std::string outFilename = resultsDir + "output_" + std::filesystem::path(filename).stem().string() + ".jpg";
        cv::imwrite(outFilename, img);
    }

    // Overall metrics
    if (totalGT > 0) {
        float overallPrecision = (totalTP + totalFP) > 0 ? float(totalTP) / float(totalTP + totalFP) : 0.0f;
        float overallRecall = (totalTP + totalFN) > 0 ? float(totalTP) / float(totalTP + totalFN) : 0.0f;
        float overallF1 = (overallPrecision + overallRecall) > 0 ? 2.0f * overallPrecision * overallRecall / (overallPrecision + overallRecall) : 0.0f;

        metricsOut << "\n=== Overall Detection Summary ===\n";
        metricsOut << "Total Ground Truth: " << totalGT << "\n";
        metricsOut << "Total True Positives: " << totalTP << "\n";
        metricsOut << "Total False Positives: " << totalFP << "\n";
        metricsOut << "Total False Negatives: " << totalFN << "\n";
        metricsOut << "Overall Precision: " << overallPrecision << "\n";
        metricsOut << "Overall Recall: " << overallRecall << "\n";
        metricsOut << "Overall F1-score: " << overallF1 << "\n";
    }

    metricsOut.close();
    return 0;
}