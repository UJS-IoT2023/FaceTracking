/**
 * src/tracker.cpp
 * @version 1.0 2026-01-03
 */
#include "tracker.h"

SharedTrack::SharedTrack(DetectionBox detection_box, int id) {
    track_id = id;
    score = detection_box.score;
    bbox = detection_box.bbox;
    frames_since_update = 0;

    kalman.init(4, 4, 0);
    kalman.transitionMatrix = (cv::Mat_<float>(4, 4) << 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
    setIdentity(kalman.measurementMatrix);
    setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-2));
    setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(1e-2));
    setIdentity(kalman.errorCovPost, cv::Scalar::all(1));

    kalman.statePost = (cv::Mat_<float>(4, 1) << (float)bbox.x, (float)bbox.y, (float)bbox.width, (float)bbox.height);
}

void SharedTrack::predict() {
    cv::Mat prediction = kalman.predict();
    bbox.x = static_cast<int>(prediction.at<float>(0));
    bbox.y = static_cast<int>(prediction.at<float>(1));
    bbox.width = static_cast<int>(prediction.at<float>(2));
    bbox.height = static_cast<int>(prediction.at<float>(3));
    frames_since_update++;
}

void SharedTrack::update(DetectionBox detection_box) {
    cv::Mat measurement = (cv::Mat_<float>(4, 1) <<
        (float)detection_box.bbox.x,
        (float)detection_box.bbox.y,
        (float)detection_box.bbox.width,
        (float)detection_box.bbox.height);
    kalman.correct(measurement);
    bbox = detection_box.bbox;
    score = detection_box.score;
    frames_since_update = 0;
}

SortTracker::SortTracker(int max_age, float iou_threshold)
    : max_age(max_age), iou_threshold(iou_threshold) {}

double SortTracker::get_iou(cv::Rect r1, cv::Rect r2) {
    int interArea = (r1 & r2).area();
    int unionArea = r1.area() + r2.area() - interArea;
    if (unionArea <= 0) return 0;
    return static_cast<double>(interArea) / unionArea;
}

std::vector<SharedTrack> SortTracker::update(const std::vector<DetectionBox>& detections) {
    for (auto& track : tracks) {
        track.predict();
    }

    std::vector<bool> det_used(detections.size(), false);
    std::vector<bool> track_updated(tracks.size(), false);

    for (size_t i = 0; i < tracks.size(); ++i) {
        double best_iou = -1.0;
        int best_det_idx = -1;

        for (size_t j = 0; j < detections.size(); ++j) {
            if (det_used[j]) continue;

            double iou = get_iou(tracks[i].bbox, detections[j].bbox);
            if (iou > iou_threshold && iou > best_iou) {
                best_iou = iou;
                best_det_idx = static_cast<int>(j);
            }
        }

        if (best_det_idx != -1) {
            tracks[i].update(detections[best_det_idx]);
            det_used[best_det_idx] = true;
            track_updated[i] = true;
        }
    }

    for (size_t i = 0; i < detections.size(); ++i) {
        if (!det_used[i] && detections[i].score > 0.5) {
            tracks.emplace_back(detections[i], next_id++);
        }
    }

    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
        [this](const SharedTrack& t) {
            return t.frames_since_update > max_age;
        }), tracks.end());

    return tracks;
}