// TfliteImageClassifier.cpp
//
// Rewritten from Exercise 5's TfliteGestureClassifier.
// See TfliteImageClassifier.h for full change notes.

#include "TfliteImageClassifier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

// ---------------------------------------------------------------------------
// Internal helpers (unchanged from Exercise 5)
// ---------------------------------------------------------------------------

namespace {

bool IsSupportedTensorType(TfLiteType type) {
    return type == kTfLiteFloat32 || type == kTfLiteInt8;
}

const char* TensorTypeName(TfLiteType type) {
    switch (type) {
        case kTfLiteFloat32: return "float32";
        case kTfLiteInt8:    return "int8";
        default:             return "unsupported";
    }
}

// Quantize a float in [-1, 1] to int8 using the tensor's scale/zero_point.
int8_t QuantizeInt8(float value, float scale, int zero_point) {
    int q = static_cast<int>(std::lround(value / scale + static_cast<float>(zero_point)));
    if (q < static_cast<int>(std::numeric_limits<int8_t>::min()))
        q = static_cast<int>(std::numeric_limits<int8_t>::min());
    if (q > static_cast<int>(std::numeric_limits<int8_t>::max()))
        q = static_cast<int>(std::numeric_limits<int8_t>::max());
    return static_cast<int8_t>(q);
}

// Dequantize int8 back to float.
float DequantizeInt8(int8_t value, float scale, int zero_point) {
    return scale * static_cast<float>(static_cast<int>(value) - zero_point);
}

} // namespace

// ---------------------------------------------------------------------------
// Impl struct
// ---------------------------------------------------------------------------

struct TfliteImageClassifier::Impl {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter>     interpreter;
};

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

TfliteImageClassifier::TfliteImageClassifier(const std::string& model_path,
                                             int num_classes)
    : impl_(std::make_unique<Impl>()), num_classes_(num_classes)
{
    ok_ = Load(model_path);
}

TfliteImageClassifier::~TfliteImageClassifier() = default;

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

bool TfliteImageClassifier::Load(const std::string& model_path) {
    // Load .tflite flatbuffer from disk.
    impl_->model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!impl_->model) {
        error_message_ = "Failed to load model: " + model_path;
        return false;
    }

    // Build interpreter.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*impl_->model, resolver);
    if (builder(&impl_->interpreter) != kTfLiteOk || !impl_->interpreter) {
        error_message_ = "Failed to create TFLite interpreter.";
        return false;
    }

    // Single CPU thread — adding more threads hurts on RPi 3 due to cache
    // contention, and we need predictable latency for the FPS requirement.
    impl_->interpreter->SetNumThreads(1);

    if (impl_->interpreter->AllocateTensors() != kTfLiteOk) {
        error_message_ = "Failed to allocate tensors.";
        return false;
    }

    // Log success.
    std::cout << "Model loaded successfully: "
          << model_path << std::endl;

    // Exactly one input and one output.
    if (impl_->interpreter->inputs().size() != 1) {
        error_message_ = "Model must have exactly one input tensor.";
        return false;
    }
    if (impl_->interpreter->outputs().size() != 1) {
        error_message_ = "Model must have exactly one output tensor.";
        return false;
    }

    const TfLiteTensor* input  = impl_->interpreter->input_tensor(0);
    const TfLiteTensor* output = impl_->interpreter->output_tensor(0);

    // Validate input type.
    if (input == nullptr || !IsSupportedTensorType(input->type)) {
        error_message_ = "Input tensor must be float32 or int8; got ";
        error_message_ += (input == nullptr) ? "null" : TensorTypeName(input->type);
        return false;
    }

    // Validate input shape: expect [1, H, W, C] — a single image with 3 channels.
    // The original Exercise 5 code accepted any flat shape; we are stricter here
    // because an image model with the wrong input shape is a silent bug.
    if (input->dims == nullptr || input->dims->size != 4) {
        error_message_ = "Expected 4D input tensor [batch, height, width, channels].";
        return false;
    }
    if (input->dims->data[3] != 3) {
        error_message_ = "Expected 3-channel (RGB) input tensor.";
        return false;
    }

    // Validate output type.
    if (output == nullptr || !IsSupportedTensorType(output->type)) {
        error_message_ = "Output tensor must be float32 or int8; got ";
        error_message_ += (output == nullptr) ? "null" : TensorTypeName(output->type);
        return false;
    }

    // Validate output shape: expect [1, num_classes].
    // This replaces the hardcoded "!= 4" check from Exercise 5.
    if (output->dims == nullptr ||
        output->dims->size != 2 ||
        output->dims->data[1] != num_classes_)
    {
        error_message_ = "Expected output shape [1, " + std::to_string(num_classes_) +
                         "]; got [1, " +
                         (output->dims ? std::to_string(output->dims->data[1]) : "?") + "].";
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// CopyInput
// ---------------------------------------------------------------------------

bool TfliteImageClassifier::CopyInput(const std::vector<float>& input_data) {
    const TfLiteTensor* input_tensor = impl_->interpreter->input_tensor(0);

    // Compute expected element count from tensor dims.
    int expected = 1;
    for (int i = 0; i < input_tensor->dims->size; ++i) {
        expected *= input_tensor->dims->data[i];
    }

    // Size mismatch is a hard error for image models — unlike IMU data we do
    // NOT silently pad/crop, because that would corrupt the spatial structure.
    if (static_cast<int>(input_data.size()) != expected) {
        error_message_ = "Input size mismatch: expected " + std::to_string(expected) +
                         " values, got " + std::to_string(input_data.size()) + ".";
        return false;
    }

    // float32 model: copy directly.
    if (input_tensor->type == kTfLiteFloat32) {
        float* dst = impl_->interpreter->typed_input_tensor<float>(0);
        std::copy(input_data.begin(), input_data.end(), dst);
        return true;
    }

    // int8 model: quantize each float value.
    if (input_tensor->type == kTfLiteInt8) {
        int8_t* dst = impl_->interpreter->typed_input_tensor<int8_t>(0);
        const float scale      = input_tensor->params.scale;
        const int   zero_point = input_tensor->params.zero_point;
        for (std::size_t i = 0; i < input_data.size(); ++i) {
            dst[i] = QuantizeInt8(input_data[i], scale, zero_point);
        }
        return true;
    }

    error_message_ = "Unsupported input tensor type (should have been caught in Load).";
    return false;
}

// ---------------------------------------------------------------------------
// ReadOutput
// ---------------------------------------------------------------------------

std::vector<float> TfliteImageClassifier::ReadOutput() const {
    const TfLiteTensor* output_tensor = impl_->interpreter->output_tensor(0);

    // float32 model: read directly.
    if (output_tensor->type == kTfLiteFloat32) {
        const float* src = impl_->interpreter->typed_output_tensor<float>(0);
        return std::vector<float>(src, src + num_classes_);
    }

    // int8 model: dequantize.
    std::vector<float> scores(static_cast<std::size_t>(num_classes_), 0.f);
    if (output_tensor->type == kTfLiteInt8) {
        const int8_t* src  = impl_->interpreter->typed_output_tensor<int8_t>(0);
        const float   scale      = output_tensor->params.scale;
        const int     zero_point = output_tensor->params.zero_point;
        for (int i = 0; i < num_classes_; ++i) {
            scores[static_cast<std::size_t>(i)] = DequantizeInt8(src[i], scale, zero_point);
        }
    }
    return scores;
}

// ---------------------------------------------------------------------------
// Predict
// ---------------------------------------------------------------------------

ImagePrediction TfliteImageClassifier::Predict(const std::vector<float>& input_data) {
    ImagePrediction pred;

    if (!ok_) {
        return pred;
    }

    if (!CopyInput(input_data)) {
        ok_ = false;
        return pred;
    }

    if (impl_->interpreter->Invoke() != kTfLiteOk) {
        error_message_ = "TFLite inference failed.";
        ok_ = false;
        return pred;
    }

    pred.scores = ReadOutput();

    const auto best = std::max_element(pred.scores.begin(), pred.scores.end());
    pred.predicted_class = static_cast<int>(
        std::distance(pred.scores.begin(), best));
    pred.confidence = *best;

    return pred;
}
