#pragma once

// preprocess_accessory.h
//
// Preprocessing for the ACCESSORY model only.
// Both models now use MobileNetV2 — preprocessing is identical.
//
// Gesture model (preprocess.h):    MobileNetV2,  pixel -> [-1, 1]
// Accessory model (this file):     MobileNetV2,  pixel -> [-1, 1]
//
// RULE: this file must match train_accessory.py preprocess():
//     Python:  (X / 127.5) - 1.0
//     C++:     pixel / 127.5f - 1.0f

#include <cstdint>
#include <vector>
#include "RpiCameraCapture.hpp"

namespace preprocess_accessory {

constexpr int IMAGE_WIDTH    = 224;
constexpr int IMAGE_HEIGHT   = 224;
constexpr int IMAGE_CHANNELS = 3;
constexpr int IMAGE_ELEMENTS = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS;

// Converts a raw RGB frame to float input for the accessory model.
// Pixel values mapped [0, 255] -> [-1.0, 1.0] (MobileNetV2 standard).
inline std::vector<float> frame_to_float(const rpicam::RgbFrame& frame) {
    std::vector<float> out;
    out.reserve(static_cast<std::size_t>(IMAGE_ELEMENTS));
    for (std::size_t i = 0; i < frame.rgb.size(); ++i) {
        out.push_back(static_cast<float>(frame.rgb[i]) / 127.5f - 1.0f);
    }
    return out;
}

} // namespace preprocess_accessory
