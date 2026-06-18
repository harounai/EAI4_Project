#include <string>
#include <thread>
#include <chrono>
#include <filesystem>
#include "RpiCameraCapture.hpp"
#include "RgbFrameBmpExport.hpp"
#include <iostream>

int main(int argc, char **argv)
{
    std::string label  = (argc > 1) ? argv[1] : "neutral";
    std::string member = (argc > 2) ? argv[2] : "";
    std::string run    = (argc > 3) ? argv[3] : "";

    std::string dir = "dataset/" + label + "/";
    std::filesystem::create_directories(dir);

    // Continue numbering from existing files
    int frame_id = 0;
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() != ".bmp") continue;
        const std::string stem = entry.path().stem().string();
        const auto pos = stem.rfind('_');
        if (pos == std::string::npos) continue;
        try {
            int n = std::stoi(stem.substr(pos + 1));
            if (n + 1 > frame_id) frame_id = n + 1;
        } catch (...) {}
    }

    rpicam::CaptureParameters params;
    rpicam::RpiCameraCapture camera(params);

    int collected = 0;

    while (true)
    {
        std::cout
        << "\nPose: "
        << label
        << "\nPress ENTER when ready or 'exit' to quit\n";

        std::string input;
        std::getline(std::cin, input);

        if (input == "exit")
            break;

        auto frame = camera.currentFrame();

        if (!frame)
        {
            std::cout << "No frame available\n";
            continue;
        }

        std::string filename =
            dir +
            label + "_" +
            member + "_" +
            run + "_" +
            std::to_string(frame_id++) +
            ".bmp";

        rpicam::saveRgbFrameAsBmp(frame, filename);

        ++collected;

        std::cout
            << "Saved: "
            << filename
            << std::endl;
    }
}
