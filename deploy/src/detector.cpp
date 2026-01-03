/**
 * src/face_detector.cpp
 * @version 1.0 2026-01-03
 * @author cacc
 */
#include "detector.h"
#include <fstream>
#include <torch/cuda.h>

Detector::Detector(const std::string &model_path) {
    try {
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available! Using GPU: " << torch::cuda::device_count() << std::endl;
            device = torch::kCUDA;
        } else {
            std::cout << "CUDA not found! Using CPU." << std::endl;
            device = torch::kCPU;
        }

        std::ifstream is(model_path, std::ios::binary);
        if (!is) throw std::runtime_error("Cannot open model file: " + model_path);

        module = torch::jit::load(is, device);
        module.eval();

        // Warmup (Crucial for GPU initialization)
        torch::NoGradGuard no_grad;
        auto warmup_input = torch::zeros({1, 3, 640, 640}).to(device);
        module.forward({warmup_input});

        std::cout << "Model loaded and warmed up on " << (device.is_cuda() ? "GPU" : "CPU") << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        exit(-1);
    }
}

std::vector<DetectionBox> Detector::inference(cv::Mat &frame) {
    if (frame.empty()) return {};

    // 1. Pre-processing (CPU)
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(640, 640));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // 2. Data Upload to GPU
    // .clone() is mandatory to ensure the Mat memory isn't corrupted during async GPU tasks
    torch::Tensor input_tensor = torch::from_blob(resized.data, {1, 640, 640, 3}, torch::kByte).clone();
    input_tensor = input_tensor.permute({0, 3, 1, 2}).to(device).to(torch::kFloat).div(255.0);

    // 3. GPU Inference
    torch::NoGradGuard no_grad;
    torch::Tensor output;
    try {
        output = module.forward({input_tensor}).toTensor();
    } catch (const std::exception &e) {
        std::cerr << "Forward Error: " << e.what() << std::endl;
        return {};
    }

    // 4. Move to CPU and Sync
    // Crossing from GPU to CPU is where 0xC0000005 usually happens.
    // We move the tensor back to CPU before touching its internal pointers.
    output = output.to(torch::kCPU).squeeze().detach();

    if (output.dim() < 2) return {};

    // Handle YOLOv8 shape [84, 8400] -> [8400, 84]
    if (output.size(0) < output.size(1)) {
        output = output.transpose(0, 1);
    }
    output = output.contiguous(); // Ensure data is linear for pointer access

    const int anchors = output.size(0);
    const int channels = output.size(1);
    const float *data_ptr = output.data_ptr<float>();

    std::vector<cv::Rect> bboxes;
    std::vector<float> confs;

    for (int i = 0; i < anchors; ++i) {
        const float *row = data_ptr + (i * channels);

        float score = 0.0f;
        if (channels > 4) {
            // For YOLOv8/v10, indices 4 to end are class scores
            for (int j = 4; j < channels; ++j) {
                if (row[j] > score) score = row[j];
            }
        }

        if (score > 0.45f) {
            float cx = row[0], cy = row[1], w = row[2], h = row[3];
            int left = static_cast<int>(cx - w / 2.0f);
            int top = static_cast<int>(cy - h / 2.0f);
            bboxes.push_back(cv::Rect(left, top, static_cast<int>(w), static_cast<int>(h)));
            confs.push_back(score);
        }
    }

    if (bboxes.empty()) return {};

    // 5. NMS (OpenCV side)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, confs, 0.45f, 0.5f, indices);

    // 6. Final Scaling
    std::vector<DetectionBox> results;
    float scale_x = (float) frame.cols / 640.0f;
    float scale_y = (float) frame.rows / 640.0f;

    for (int idx: indices) {
        DetectionBox det;
        det.bbox.x = static_cast<int>(bboxes[idx].x * scale_x);
        det.bbox.y = static_cast<int>(bboxes[idx].y * scale_y);
        det.bbox.width = static_cast<int>(bboxes[idx].width * scale_x);
        det.bbox.height = static_cast<int>(bboxes[idx].height * scale_y);
        det.score = confs[idx];
        results.push_back(det);
    }
    return results;
}
