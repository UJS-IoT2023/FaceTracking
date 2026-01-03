//
// Created by cacc on 1/3/26.
//

#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

struct DetectionBox {
    cv::Rect bbox;
    float score;
};

class FaceDetector {

public:
    explicit FaceDetector(const std::string& model_path);
    std::vector<DetectionBox> inference(cv::Mat& frame);

private:
    torch::jit::script::Module module;
    torch::Device device = torch::kCUDA;
};
#endif
