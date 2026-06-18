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

    rpicam::RpiCameraCapture camera(params);

    TfliteImageClassifier gesture_classifier(
        gesture::GESTURE_MODEL_PATH,
        gesture::NUM_GESTURE_CLASSES);

    TfliteImageClassifier accessory_classifier(
        gesture::ACCESSORY_MODEL_PATH,
        gesture::NUM_ACCESSORY_CLASSES);

    SenseHatDisplay display;

while (true)
{

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

    auto frame = camera.currentFrame();

    if (!frame)
    {
        display.ShowErrorMarker();
        return 1;
    }

    auto gesture_input =
        preprocess::frame_to_float(*frame);

    auto gesture_prediction =
        gesture_classifier.Predict(gesture_input);

    auto accessory_input =
        preprocess_accessory::frame_to_float(*frame);

    auto accessory_prediction =
        accessory_classifier.Predict(accessory_input);

    Gesture player_gesture = Gesture::NEUTRAL;

    switch (gesture_prediction.predicted_class)
    {
        case gesture::ROCK:
            player_gesture = Gesture::ROCK;
            break;

        case gesture::PAPER:
            player_gesture = Gesture::PAPER;
            break;

        case gesture::SCISSORS:
            player_gesture = Gesture::SCISSORS;
            break;

        default:
            player_gesture = Gesture::NEUTRAL;
            break;
    }

    bool accessory_present =
        accessory_prediction.predicted_class ==
        gesture::ACCESSORY_PRESENT; // should be 1 if accessory is present, 0 otherwise

    Gesture pi_gesture =
        DeterminePiGesture(
            player_gesture,
            accessory_present);

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
