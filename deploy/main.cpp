/**
 * main.cpp
 * The entry of the program
 * @version 1.0 2026-01-03
 * @author cacc
 */
#include "detector.h"
#include "tracker.h"

const std::string MODEL_PATH = "/home/cacc/Workspace/FaceTracking/model/best.torchscript";
const int MAX_AGE = 15;
const float IOU_THRESHOLD = 0.3;

int main() {
    Detector detector(MODEL_PATH);
    SortTracker tracker(MAX_AGE, IOU_THRESHOLD);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Camera Error!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        auto detections = detector.inference(frame);

        auto online_tracks = tracker.update(detections);

        for (const auto& track : online_tracks) {
            cv::rectangle(frame, track.bbox, cv::Scalar(0, 255, 0), 2);
            std::string text = "ID: " + std::to_string(track.track_id);
            cv::putText(frame, text, cv::Point(track.bbox.x, track.bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        }

        cv::imshow("Face Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}