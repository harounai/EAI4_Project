#include "RpiCameraCapture.hpp"
#include "RgbFrameBmpExport.hpp"
#include "TfliteImageClassifier.h"
#include "sense_hat_display.h"
#include "game_logic.h"
#include "model_classes.h"
#include "preprocess.h"
#include "preprocess_accessory.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

rpicam::CaptureParameters params;

const char* GestureName(int idx)
{
    switch (idx)
    {
        case 0: return "NEUTRAL";
        case 1: return "PAPER";
        case 2: return "ROCK";
        case 3: return "SCISSORS";
        default: return "UNKNOWN";
    }
}

int main()
{
    // try {
    //     rpicam::CaptureParameters params;
    //     params.width = 224;
    //     params.height = 224;

    //     params.shutter_us = 0;      // set for manual shutter, e.g. 8000
    //     params.gain = 1.0f;         // increase for dark environments
    //     params.buffer_count = 6;
    //     params.awb = true;

    //     rpicam::RpiCameraCapture camera(params);

    //     while (true) {
    //         // Get current image to framebuffer.
    //         std::shared_ptr<const rpicam::RgbFrame> frame = camera.currentFrame();
    //         if (!frame) {
    //             std::this_thread::sleep_for(std::chrono::microseconds(100));
    //             continue;
    //         }

    //         rpicam::saveRgbFrameAsBmp(frame, "frame.bmp");
    //         std::cout << "Saved frame.bmp from sequence " << frame->sequence << "\n";

    //         break; // store one BMP and exit
    //     }

    //     return 0;
    // } catch (const std::exception &e) {
    //     std::cerr << "Error: " << e.what() << "\n";
    //     return 1;
    // }

    // helper class to convert gesture index to string for logging

    rpicam::RpiCameraCapture camera(params);

    TfliteImageClassifier gesture_classifier(
        gesture::GESTURE_MODEL_PATH,
        gesture::NUM_GESTURE_CLASSES);

    std::cout << "Gesture classifier loaded successfully." << std::endl;

    TfliteImageClassifier accessory_classifier(
        gesture::ACCESSORY_MODEL_PATH,
        gesture::NUM_ACCESSORY_CLASSES);

    std::cout << "Accessory classifier loaded successfully." << std::endl;

    SenseHatDisplay display;

    std::cout << "Display created" << std::endl;

    std::cout << "Entering loop" << std::endl;

while (true)
{
    std::cout << "Starting new round..." << std::endl;

    display.ShowCountdownDigit(3);
    std::this_thread::sleep_for(
    std::chrono::seconds(1));

    display.ShowCountdownDigit(2);
    std::this_thread::sleep_for(
    std::chrono::seconds(1));

    display.ShowCountdownDigit(1);
    std::this_thread::sleep_for(
    std::chrono::seconds(1));

    display.Clear();

    display.FillBlue();
    std::this_thread::sleep_for(
    std::chrono::milliseconds(200));

    std::cout << "Before currentFrame()" << std::endl;

    auto frame = camera.currentFrame();

    std::cout << "After currentFrame()" << std::endl;

    // Check if the frame is valid
    if (!frame)
    {
        display.ShowErrorMarker();
        // std::cerr << "Error: Failed to capture frame." << std::endl;
        return 1;
    }
    else
    {
        std::cout << "Captured frame successfully!" << std::endl;
    }

    auto gesture_input =
        preprocess::frame_to_float(*frame);
    // Check if the gesture input is valid
    if (gesture_input.empty())
    {
        display.ShowErrorMarker();
        std::cerr << "Error: Gesture input is empty." << std::endl;
        return 1;
    }

    auto gesture_prediction =
        gesture_classifier.Predict(gesture_input);
    // Check if the gesture prediction is valid
    if (gesture_prediction.predicted_class < 0 || gesture_prediction.predicted_class >= gesture::NUM_GESTURE_CLASSES)
    {
        display.ShowErrorMarker();
        std::cerr << "Error: Invalid gesture prediction class index: "
                  << gesture_prediction.predicted_class << std::endl;
        return 1;
    }

    auto accessory_input =
        preprocess_accessory::frame_to_float(*frame);
    // Check if the accessory input is valid
    if (accessory_input.empty())
    {
        display.ShowErrorMarker();
        std::cerr << "Error: Accessory input is empty." << std::endl;
        return 1;
    }

    auto accessory_prediction =
        accessory_classifier.Predict(accessory_input);
    // Check if the accessory prediction is valid
    if (accessory_prediction.predicted_class < 0 || accessory_prediction.predicted_class >= gesture::NUM_ACCESSORY_CLASSES)
    {
        display.ShowErrorMarker();
        std::cerr << "Error: Invalid accessory prediction class index: "
                  << accessory_prediction.predicted_class << std::endl;
        return 1;
    }

    Gesture player_gesture = Gesture::NEUTRAL;

    switch (gesture_prediction.predicted_class)
    {
        case gesture::ROCK:
            player_gesture = Gesture::ROCK; // index 2
            break;

        case gesture::PAPER:
            player_gesture = Gesture::PAPER; // index 1
            break;

        case gesture::SCISSORS:
            player_gesture = Gesture::SCISSORS; // index 3
            break;

        default:
            player_gesture = Gesture::NEUTRAL; // index 0
            break;
    }

    bool accessory_present =
        accessory_prediction.predicted_class ==
        gesture::ACCESSORY_PRESENT; // should be 1 if accessory is present, 0 otherwise

    Gesture pi_gesture =
        DeterminePiGesture(
            player_gesture,
            accessory_present);

    // Log the predictions and accessory presence
    std::cout
    << "Human gesture = "
    << GestureName(gesture_prediction.predicted_class)
    << " (" << gesture_prediction.predicted_class << ")\n"

    << "Accessory = "
    << (accessory_present ? "PRESENT" : "ABSENT")
    << " (" << accessory_prediction.predicted_class << ")\n"

    << "Pi gesture = "
    << GestureName(static_cast<int>(pi_gesture))
    << " (" << static_cast<int>(pi_gesture) << ")\n"

    << "Outcome = "
    << (ComputeOutcome(accessory_present) == Outcome::WIN
            ? "WIN"
            : "LOSE")
    << "\n"

    << "-------------------------------------------------\n";

    // Display Pi's gesture on Sense HAT
    switch (pi_gesture)
    {
        case Gesture::ROCK:
            display.ShowRock();
            break;
        case Gesture::PAPER:
            display.ShowPaper();
            break;
        case Gesture::SCISSORS:
            display.ShowScissors();
            break;
        default:
            display.ShowErrorMarker();
            break;
    }

    std::this_thread::sleep_for(
    std::chrono::seconds(3));

    Outcome outcome =
        ComputeOutcome(
            accessory_present);

    if (outcome == Outcome::WIN)
    {
        display.FillGreen(); // Human won
        std::this_thread::sleep_for(
        std::chrono::seconds(3));
        display.Clear();
    }
    else
    {
        display.FillRed(); // Human lost
        std::this_thread::sleep_for(
        std::chrono::seconds(3));
        display.Clear();
    }
}

}
