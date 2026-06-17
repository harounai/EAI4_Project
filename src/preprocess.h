#pragma once

// preprocess.h
//
// Shared preprocessing contract between the ML models (P1/P2) and the C++
// inference engine (P3).
//
// RULE: whatever this file does in C++ must exactly match what
//       train_gesture.py does in Python. If one changes, both must change.
//
// Current contract:
//   Python (train_gesture.py):   X = (pixel / 127.5) - 1.0
//   C++ (this file):             val = (pixel / 127.5f) - 1.0f
//
//   Input range:  [0, 255]  (uint8 RGB from RpiCameraCapture)
//   Output range: [-1, 1]   (MobileNetV2 standard)
//   Channel order: RGB       (RpiCameraCapture already outputs RGB888, no swap needed)

#include <cstdint>
#include <vector>
#include "RpiCameraCapture.hpp"  // for rpicam::RgbFrame

namespace preprocess {

// Image dimensions — must match CaptureParameters in main.cpp.
constexpr int IMAGE_WIDTH    = 224;
constexpr int IMAGE_HEIGHT   = 224;
constexpr int IMAGE_CHANNELS = 3;
constexpr int IMAGE_ELEMENTS = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS; // 150528

// Converts a raw RGB frame from RpiCameraCapture into a float vector
// suitable for TfliteImageClassifier::Predict().
//
// Pixel values are mapped [0, 255] -> [-1.0, 1.0] (MobileNetV2 standard).
// Channel order is preserved as RGB (no BGR swap needed).
//
// The returned vector has IMAGE_ELEMENTS = 224*224*3 = 150528 floats,
// laid out as [R00, G00, B00, R01, G01, B01, ..., R(H-1)(W-1), G..., B...].
inline std::vector<float> frame_to_float(
    const rpicam::RgbFrame& frame)
{
    std::vector<float> out;
    out.reserve(static_cast<std::size_t>(IMAGE_ELEMENTS));

    for (std::size_t i = 0; i < frame.rgb.size(); ++i) {
        out.push_back(
            static_cast<float>(frame.rgb[i]) / 127.5f - 1.0f
        );
    }

    return out;
}

} // namespace preprocess
