#pragma once

// TfliteImageClassifier.h
//
// Rewritten from Exercise 5's TfliteGestureClassifier to work with image
// models (224x224 RGB input) instead of IMU time-series.
//
// Key changes from the original:
//  - Output class count is now a constructor parameter, not hardcoded to 4.
//    This lets you use the same class for both the gesture model (4 classes)
//    and the accessory model (2 classes).
//  - Input validation checks for a 4D image tensor [1, H, W, C] instead of
//    the flat 1D IMU tensor.
//  - The hardcoded "Expected 4 output classes" check is gone.
//  - Class name is updated to reflect the new purpose.
//
// Usage:
//   TfliteImageClassifier gesture_clf("gesture_model.tflite", 4);
//   TfliteImageClassifier accessory_clf("accessory_model.tflite", 2);
//
//   auto pred = gesture_clf.Predict(input_floats);  // input_floats = preprocessed image

#include <memory>
#include <string>
#include <vector>

struct ImagePrediction {
    int   predicted_class = -1;  // index of highest-scoring class
    float confidence      = 0.f; // score of the winning class
    std::vector<float> scores;   // all class scores, size == num_classes
};

class TfliteImageClassifier {
public:
    // model_path:  path to the .tflite file
    // num_classes: expected number of output classes — must match the model
    explicit TfliteImageClassifier(const std::string& model_path, int num_classes);
    ~TfliteImageClassifier();

    // Non-copyable, non-movable (holds OS resources via TFLite interpreter)
    TfliteImageClassifier(const TfliteImageClassifier&)            = delete;
    TfliteImageClassifier& operator=(const TfliteImageClassifier&) = delete;
    TfliteImageClassifier(TfliteImageClassifier&&)                 = delete;
    TfliteImageClassifier& operator=(TfliteImageClassifier&&)      = delete;

    bool               ok()            const { return ok_; }
    const std::string& error_message() const { return error_message_; }
    int                num_classes()   const { return num_classes_; }

    // Run inference on a preprocessed image.
    // input_data: flattened float vector of length H * W * C (= 224*224*3 = 150528).
    //             Values must be in [-1, 1] (MobileNetV2 standard preprocessing).
    //             See preprocess.h for the C++ helper that produces this.
    ImagePrediction Predict(const std::vector<float>& input_data);

private:
    struct Impl;

    bool              Load(const std::string& model_path);
    bool              CopyInput(const std::vector<float>& input_data);
    std::vector<float> ReadOutput() const;

    std::unique_ptr<Impl> impl_;
    int         num_classes_  = 0;
    bool        ok_           = false;
    std::string error_message_;
};
