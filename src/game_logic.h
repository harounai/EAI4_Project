#pragma once
#include "model_classes.h"

enum class Gesture {
    NEUTRAL  = gesture::NEUTRAL,
    PAPER    = gesture::PAPER,
    ROCK     = gesture::ROCK,
    SCISSORS = gesture::SCISSORS
};

enum class Outcome {
    WIN,
    LOSE
};

Gesture DeterminePiGesture(
    Gesture player_gesture,
    bool accessory_present);

Outcome ComputeOutcome(bool accessory_present);