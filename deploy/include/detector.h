/**
 * include/detector.h
 * FaceDetector class
 * @version 1.0 2026-01-03
 * @author cacc
 */
#ifndef DETECTOR_H
#define DETECTOR_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

struct DetectionBox {
    cv::Rect bbox;
    float score;
};

class Detector {
public:
    explicit Detector(const std::string& model_path);
    std::vector<DetectionBox> inference(cv::Mat& frame);

private:
    torch::jit::script::Module module;
    torch::Device device = torch::kCUDA;
};
#endif
