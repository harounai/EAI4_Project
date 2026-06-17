#pragma once

// model_classes.h
//
// Shared class index constants for the gesture model.
// Generated from train_gesture.py CLASS_NAMES = ["neutral", "paper", "rock", "scissors"]
// Order is alphabetical — set by image_dataset_from_directory in TensorFlow.
//
// DO NOT change these indices without retraining the gesture model.
// If you retrain with different folder names, run check_dataset.py and
// update this file to match the printed class mapping.
//
// Used by: P3 (main.cpp game logic), P4 (ShowGesture index mapping)

namespace gesture {

// Gesture model output indices (4 classes)
constexpr int NEUTRAL  = 0;
constexpr int PAPER    = 1;
constexpr int ROCK     = 2;
constexpr int SCISSORS = 3;
constexpr int NUM_GESTURE_CLASSES = 4;

// Accessory model output indices (2 classes)
// Order: absent=0, present=1
// (alphabetical: "absent" < "present")
constexpr int ACCESSORY_ABSENT  = 0;
constexpr int ACCESSORY_PRESENT = 1;
constexpr int NUM_ACCESSORY_CLASSES = 2;

// Model file paths — must match what P4 deploys to the Pi
constexpr const char* GESTURE_MODEL_PATH  = "gesture_model.tflite";
constexpr const char* ACCESSORY_MODEL_PATH = "accessory_model.tflite";

// Game logic helpers
// Returns the gesture index that BEATS the given player gesture.
// Pi uses this when the player has NO accessory (player should lose).
inline int winningCounterFor(int player_gesture) {
    switch (player_gesture) {
        case ROCK:     return PAPER;     // paper beats rock
        case PAPER:    return SCISSORS;  // scissors beats paper
        case SCISSORS: return ROCK;      // rock beats scissors
        default:       return ROCK;      // fallback
    }
}

// Returns the gesture index that LOSES TO the given player gesture.
// Pi uses this when the player HAS the accessory (player should win).
inline int losingCounterFor(int player_gesture) {
    switch (player_gesture) {
        case ROCK:     return SCISSORS;  // scissors loses to rock
        case PAPER:    return ROCK;      // rock loses to paper
        case SCISSORS: return PAPER;     // paper loses to scissors
        default:       return SCISSORS;  // fallback
    }
}

} // namespace gesture
