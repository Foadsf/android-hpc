# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.22)
# Single source of the version: the native report, hpc_cli and the APK (app/build.gradle reads it).
project(android_hpc VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

enable_testing()  # must be at the root so `ctest --test-dir <build>` finds the tests in core/
add_subdirectory(core)

if(ANDROID)
  add_library(hpcjni SHARED jni/hpc_jni.cpp)
  target_link_libraries(hpcjni PRIVATE hpc_core log)
  # 16 KiB page-size compatibility (Android 15+ devices)
  target_link_options(hpcjni PRIVATE -Wl,-z,max-page-size=16384)
endif()
