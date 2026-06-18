cmake_minimum_required(VERSION 3.20)

set(APP_NAME "mnist_sensehat_demo" CACHE STRING "Application target name")
project(${APP_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TFLITE_SRC_DIR "${CMAKE_SOURCE_DIR}/third_party/tensorflow-src" CACHE PATH "TensorFlow source tree")

if(NOT EXISTS "${TFLITE_SRC_DIR}/tensorflow/lite/interpreter.h")
  message(FATAL_ERROR "TensorFlow Lite source tree not found: ${TFLITE_SRC_DIR}. Run `make tflite` to fetch the source tree.")
endif()

add_subdirectory(
  "${TFLITE_SRC_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

find_package(Threads REQUIRED)

# Shared sources (no entry points)
set(SHARED_SOURCES
  src/RpiCameraCapture.cpp
  src/TfliteImageClassifier.cpp
  src/bmp_image.cpp
  src/sense_hat_display.cpp
  src/game_logic.cpp
)

# Main inference binary
add_executable(${APP_NAME}
  ${SHARED_SOURCES}
  src/main.cpp
)

# Dataset capture binary (no TFLite needed)
add_executable(dataset_capture
  src/RpiCameraCapture.cpp
  src/bmp_image.cpp
  src/dataset_capture.cpp
)

target_include_directories(${APP_NAME} PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
)

target_include_directories(dataset_capture PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
)

target_link_libraries(${APP_NAME} PRIVATE
  tensorflow-lite
  Threads::Threads
  dl
  m
)

target_link_libraries(dataset_capture PRIVATE
  Threads::Threads
)

target_compile_options(${APP_NAME} PRIVATE
  -Wall -Wextra -Wpedantic
)

target_compile_options(dataset_capture PRIVATE
  -Wall -Wextra -Wpedantic
)