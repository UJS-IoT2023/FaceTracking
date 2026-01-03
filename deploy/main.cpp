#include "face_detector.h"

const std::string MODEL_PATH = "/home/cacc/Workspace/FaceTracking/model/best.torchscript";

int main() {
    FaceDetector detector(MODEL_PATH);

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