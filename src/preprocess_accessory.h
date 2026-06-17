#pragma once

// preprocess_accessory.h
//
// Preprocessing for the ACCESSORY model only.
// This is DIFFERENT from preprocess.h (gesture model).
//
// Gesture model (preprocess.h):    MobileNetV2,  pixel -> [-1, 1]
// Accessory model (this file):     MobileNetV3,  pixel -> [0, 1]
//
// Both models receive the same raw camera frame, but are preprocessed
// differently before being passed to their respective classifiers.
//
// RULE: this file must match train_accessory.py preprocess_mobilenetv3():
//     Python:  X / 255.0
//     C++:     pixel / 255.0f
//
// Do NOT use this header for the gesture model. Do NOT use preprocess.h
// for the accessory model. They are incompatible.

#include <cstdint>
#include <vector>
#include "RpiCameraCapture.hpp"

namespace preprocess_accessory {

constexpr int IMAGE_WIDTH    = 224;
constexpr int IMAGE_HEIGHT   = 224;
constexpr int IMAGE_CHANNELS = 3;
constexpr int IMAGE_ELEMENTS = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS;

// Converts a raw RGB frame to float input for the accessory model.
// Pixel values mapped [0, 255] -> [0.0, 1.0] (MobileNetV3 standard).
inline std::vector<float> frame_to_float(const rpicam::RgbFrame& frame) {
    std::vector<float> out;
    out.reserve(static_cast<std::size_t>(IMAGE_ELEMENTS));
    for (std::size_t i = 0; i < frame.rgb.size(); ++i) {
        out.push_back(static_cast<float>(frame.rgb[i]) / 255.0f);
    }
    return out;
}

} // namespace preprocess_accessory
