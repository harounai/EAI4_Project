#pragma once

enum class Gesture {
    ROCK,
    PAPER,
    SCISSORS,
    NEUTRAL
};

enum class Outcome {
    WIN,
    LOSE
};

Gesture DeterminePiGesture(
    Gesture player_gesture,
    bool accessory_present);

Outcome ComputeOutcome(bool accessory_present);