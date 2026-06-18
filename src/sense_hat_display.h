#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

class SenseHatDisplay {
 public:
  SenseHatDisplay();
  ~SenseHatDisplay();

  bool available() const { return available_; }
  const std::string& error_message() const { return error_message_; }
  bool ShowDigit(int digit, float confidence);
  void ShowErrorMarker();
  void Clear();
  bool ShowCountdownDigit(int digit);

  void ShowRock();
  void ShowPaper();
  void ShowScissors();

  void FillGreen();
  void FillRed();
  void FillBlue();
  

 private:
  bool OpenFramebuffer();
  void WritePattern(const std::uint8_t pattern[8], std::uint16_t color);
  std::uint16_t MakeRgb565(std::uint8_t red, std::uint8_t green, std::uint8_t blue) const;

  int file_descriptor_ = -1;
  std::size_t mapping_size_ = 0;
  unsigned char* framebuffer_ = nullptr;
  int line_length_ = 0;
  bool available_ = false;
  std::string error_message_;

  void FillColor(
    std::uint8_t r,
    std::uint8_t g,
    std::uint8_t b);

};
