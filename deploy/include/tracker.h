/**
 * include/tracker.h
 * @version 1.0 2026-01-03
 * @author cacc
 */
#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include "detector.h"

struct SharedTrack {
    cv::Rect bbox;
    float score;
    int track_id;
    int frames_since_update;
    cv::KalmanFilter kalman;

    SharedTrack(DetectionBox detection_box, int id);
    void predict();
    void update(DetectionBox detection_box);
};

class SortTracker {
public:
    SortTracker(int max_age = 30, float iou_threshold = 0.3);
    std::vector<SharedTrack> update(const std::vector<DetectionBox>& detections);

private:
    int next_id = 1;
    int max_age;
    float iou_threshold;
    std::vector<SharedTrack> tracks;

    double get_iou(cv::Rect rect1, cv::Rect rect2);
};

#endif //TRACKER_H
