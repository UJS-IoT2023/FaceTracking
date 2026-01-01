#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/cuda.h>

struct Detection {
    cv::Rect bbox;
    float score;
};

class YoloDetector {
public:
    torch::jit::script::Module module;
    torch::Device device = torch::kCPU;

    YoloDetector(const std::string& model_path) {
        try {
            // 1. Device Selection
            if (torch::cuda::is_available()) {
                std::cout << "CUDA is available! Using GPU: " << torch::cuda::device_count() << std::endl;
                device = torch::kCUDA;
            } else {
                std::cout << "CUDA not found! Falling back to CPU." << std::endl;
                device = torch::kCPU;
            }

            // 2. Load model using binary stream
            std::ifstream is(model_path, std::ios::binary);
            if (!is) throw std::runtime_error("Cannot open model file: " + model_path);

            module = torch::jit::load(is, device);
            module.eval();

            // Warmup (Crucial for GPU initialization)
            torch::NoGradGuard no_grad;
            auto warmup_input = torch::zeros({1, 3, 640, 640}).to(device);
            module.forward({warmup_input});

            std::cout << "Model loaded and warmed up on " << (device.is_cuda() ? "GPU" : "CPU") << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Initialization failed: " << e.what() << std::endl;
            exit(-1);
        }
    }

    std::vector<Detection> inference(cv::Mat& frame) {
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
        } catch (const std::exception& e) {
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
        const float* data_ptr = output.data_ptr<float>();

        std::vector<cv::Rect> bboxes;
        std::vector<float> confs;

        for (int i = 0; i < anchors; ++i) {
            const float* row = data_ptr + (i * channels);

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
        std::vector<Detection> results;
        float scale_x = (float)frame.cols / 640.0f;
        float scale_y = (float)frame.rows / 640.0f;

        for (int idx : indices) {
            Detection det;
            det.bbox.x = static_cast<int>(bboxes[idx].x * scale_x);
            det.bbox.y = static_cast<int>(bboxes[idx].y * scale_y);
            det.bbox.width = static_cast<int>(bboxes[idx].width * scale_x);
            det.bbox.height = static_cast<int>(bboxes[idx].height * scale_y);
            det.score = confs[idx];
            results.push_back(det);
        }
        return results;
    }
};

int main() {
    // Required for Windows stability
    _putenv("XNNPACK_DISABLE=1");

    YoloDetector detector("C:/Dev/workspace/FaceTracking/deploy/cmake-build-debug/best.torchscript");

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Camera Error!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        auto detections = detector.inference(frame);

        for (const auto& det : detections) {
            cv::rectangle(frame, det.bbox, cv::Scalar(0, 255, 0), 2);
            std::string text = "Face: " + std::to_string(det.score).substr(0, 4);
            cv::putText(frame, text, cv::Point(det.bbox.x, det.bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("RTX 4060 GPU Face Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}