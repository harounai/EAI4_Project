//

#include "tflite_digit_classifier.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

struct TfliteDigitClassifier::Impl {
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
};

TfliteDigitClassifier::TfliteDigitClassifier(const std::string& model_path)
    : impl_(std::make_unique<Impl>()) {
  ok_ = Load(model_path);
}

TfliteDigitClassifier::~TfliteDigitClassifier() = default;

bool TfliteDigitClassifier::Load(const std::string& model_path) {
  // We build the model from file
  impl_->model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (!impl_->model) {
    error_message_ = "Failed to load model from " + model_path;
    return false;
  }
  
  // We build the opResolver
  tflite::ops::builtin::BuiltinOpResolver resolver;
  
  // We build the interpreter
  tflite::InterpreterBuilder builder(*impl_->model, resolver);
  if (builder(&impl_->interpreter) != kTfLiteOk || !impl_->interpreter) {
    error_message_ = "Failed to build interpreter.";
    return false;
  }
  

  if (impl_->interpreter->AllocateTensors() != kTfLiteOk) {
    error_message_ = "Failed to allocate tensors.";
    return false;
  }

  if (impl_->interpreter->inputs().size() != 1) {
    error_message_ = "The model must have exactly one input tensor.";
    return false;
  }

  if (impl_->interpreter->outputs().size() != 1) {
    error_message_ = "The model must have exactly one output tensor.";
    return false;
  }

  const TfLiteTensor* input = impl_->interpreter->input_tensor(0);
  const TfLiteTensor* output = impl_->interpreter->output_tensor(0);

  if (input == nullptr || input->type != kTfLiteFloat32) {
    error_message_ = "The model input must be float32.";
    return false;
  }

  if (output == nullptr || output->type != kTfLiteFloat32) {
    error_message_ = "The model output must be float32.";
    return false;
  }

  if (input->dims == nullptr || input->dims->size != 4 ||
      input->dims->data[0] != 1 || input->dims->data[1] != 28 ||
      input->dims->data[2] != 28 || input->dims->data[3] != 1) {
    error_message_ = "Expected input shape [1, 28, 28, 1].";
    return false;
  }

  if (output->dims == nullptr || output->dims->size != 2 ||
      output->dims->data[0] != 1 || output->dims->data[1] != 10) {
    error_message_ = "Expected output shape [1, 10].";
    return false;
  }

  return true;
}

bool TfliteDigitClassifier::CopyInput(
    const std::vector<float>& normalized_image_28x28) {
  if (!impl_ || !impl_->interpreter) {
    error_message_ = "Interpreter is unavailable.";
    return false;
  }

  if (normalized_image_28x28.size() != 28U * 28U) {
    error_message_ = "Expected 28x28 input values.";
    return false;
  }

  // We transfer the image into the input buffer
  float* input = impl_->interpreter->typed_input_tensor<float>(0);
  std::copy(normalized_image_28x28.begin(), normalized_image_28x28.end(), input);
  
  return true;
}

std::vector<float> TfliteDigitClassifier::ReadOutput() const {
  const float* output = impl_->interpreter->typed_output_tensor<float>(0);
  return std::vector<float>(output, output + 10);
}

DigitPrediction TfliteDigitClassifier::Predict(
    const std::vector<float>& normalized_image_28x28) {
  DigitPrediction prediction;
  prediction.digit = -1;
  prediction.confidence = 0.0F;

  if (!ok_) {
    return prediction;
  }

  if (!CopyInput(normalized_image_28x28)) {
    ok_ = false;
    return prediction;
  }

  // We call the interpreter for invoke
  if (impl_->interpreter->Invoke() != kTfLiteOk) {
    error_message_ = "Failed to invoke interpreter.";
    return prediction;
  }

  prediction.probabilities = ReadOutput();

  const auto best = std::max_element(prediction.probabilities.begin(),
                                     prediction.probabilities.end());
  prediction.digit =
      static_cast<int>(std::distance(prediction.probabilities.begin(), best));
  prediction.confidence = *best;
  return prediction;
}
