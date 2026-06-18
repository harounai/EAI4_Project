#include "game_logic.h"
#include "model_classes.h"

Gesture DeterminePiGesture(
    Gesture player_gesture,
    bool accessory_present)
{
    if (player_gesture == Gesture::NEUTRAL)
        return Gesture::NEUTRAL;

    int player_index;

    switch (player_gesture)
    {
        case Gesture::ROCK:
            player_index = gesture::ROCK;
            break;

        case Gesture::PAPER:
            player_index = gesture::PAPER;
            break;

        case Gesture::SCISSORS:
            player_index = gesture::SCISSORS;
            break;

        default:
            player_index = gesture::ROCK;
            break;
    }

    int pi_index;

    if (accessory_present)
    {
        pi_index = gesture::losingCounterFor(player_index);
    }
    else
    {
        pi_index = gesture::winningCounterFor(player_index);
    }

    switch (pi_index)
    {
        case gesture::ROCK:
            return Gesture::ROCK;

        case gesture::PAPER:
            return Gesture::PAPER;

        case gesture::SCISSORS:
            return Gesture::SCISSORS;

        default:
            return Gesture::NEUTRAL;
    }
}

Outcome ComputeOutcome(bool accessory_present)
{
    return accessory_present
        ? Outcome::WIN
        : Outcome::LOSE;
}