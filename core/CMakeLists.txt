# SPDX-License-Identifier: Apache-2.0
option(HPC_WITH_VULKAN "Build the Vulkan backend" ON)
option(HPC_BUILD_CLI "Build hpc_cli" ON)
if(ANDROID)
  option(HPC_BUILD_TESTS "Build host tests" OFF)
else()
  option(HPC_BUILD_TESTS "Build host tests" ON)
endif()

find_package(Threads REQUIRED)

set(HPC_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${HPC_GEN_DIR})

# ---- shaders: GLSL bodies -> SPIR-V (Vulkan) and ES text (GLES) -------------------------------
set(HPC_KERNELS sgemm conv3x3 reduce triad)
set(HPC_GENERATED_INC "")
set(HPC_SPV_FILES "")

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/shaders/header_vk.glsl HDR_VK)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/shaders/header_es.glsl HDR_ES)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  shaders/header_vk.glsl shaders/header_es.glsl)

set(_vk_possible FALSE)
if(HPC_WITH_VULKAN)
  # glslc: from the NDK (Android) or the Vulkan SDK / PATH (host)
  set(_hints)
  if(ANDROID AND DEFINED ANDROID_NDK)
    list(APPEND _hints ${ANDROID_NDK}/shader-tools/${ANDROID_NDK_HOST_SYSTEM_NAME})
  endif()
  if(DEFINED ENV{VULKAN_SDK})
    list(APPEND _hints $ENV{VULKAN_SDK}/Bin $ENV{VULKAN_SDK}/bin)
  endif()
  find_program(HPC_GLSLC glslc HINTS ${_hints})
  if(NOT ANDROID)
    find_package(Vulkan QUIET)
  endif()
  if(HPC_GLSLC AND (ANDROID OR Vulkan_FOUND))
    set(_vk_possible TRUE)
  else()
    message(WARNING "Vulkan backend disabled: need glslc (${HPC_GLSLC}) and Vulkan headers (found=${Vulkan_FOUND})")
  endif()
endif()

foreach(k ${HPC_KERNELS})
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${k}.glsl BODY)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS shaders/${k}.glsl)
  # ES text (always generated; only compiled into the Android build)
  file(WRITE ${HPC_GEN_DIR}/${k}_es.glsl "${HDR_ES}\n${BODY}")
  add_custom_command(OUTPUT ${HPC_GEN_DIR}/es_${k}.inc
    COMMAND ${CMAKE_COMMAND} -DSRC=${HPC_GEN_DIR}/${k}_es.glsl -DDST=${HPC_GEN_DIR}/es_${k}.inc
            -DSYMBOL=kEs_${k} -DMODE=text -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    DEPENDS ${HPC_GEN_DIR}/${k}_es.glsl ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake VERBATIM)
  list(APPEND HPC_GENERATED_INC ${HPC_GEN_DIR}/es_${k}.inc)
  if(_vk_possible)
    file(WRITE ${HPC_GEN_DIR}/${k}.comp "${HDR_VK}\n${BODY}")
    add_custom_command(OUTPUT ${HPC_GEN_DIR}/${k}.spv
      COMMAND ${HPC_GLSLC} -fshader-stage=compute --target-env=vulkan1.0 -O -o ${HPC_GEN_DIR}/${k}.spv ${HPC_GEN_DIR}/${k}.comp
      DEPENDS ${HPC_GEN_DIR}/${k}.comp VERBATIM)
    add_custom_command(OUTPUT ${HPC_GEN_DIR}/spv_${k}.inc
      COMMAND ${CMAKE_COMMAND} -DSRC=${HPC_GEN_DIR}/${k}.spv -DDST=${HPC_GEN_DIR}/spv_${k}.inc
              -DSYMBOL=kSpv_${k} -DMODE=bin -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
      DEPENDS ${HPC_GEN_DIR}/${k}.spv ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake VERBATIM)
    list(APPEND HPC_GENERATED_INC ${HPC_GEN_DIR}/spv_${k}.inc)
    list(APPEND HPC_SPV_FILES ${HPC_GEN_DIR}/${k}.spv)
  endif()
endforeach()

# ---- library --------------------------------------------------------------------------------
set(HPC_SOURCES
  src/log.cpp src/types.cpp src/reference.cpp src/cpu.cpp src/cpu_backend.cpp src/sycl_backend.cpp
  src/opencl_backend.cpp src/gles_backend.cpp src/nnapi_backend.cpp src/discover.cpp src/bench.cpp)
if(_vk_possible)
  list(APPEND HPC_SOURCES src/vulkan_backend.cpp)
else()
  list(APPEND HPC_SOURCES src/vulkan_stub.cpp)
endif()

add_library(hpc_core STATIC ${HPC_SOURCES} ${HPC_GENERATED_INC})
target_include_directories(hpc_core PUBLIC include PRIVATE src ${HPC_GEN_DIR})
target_link_libraries(hpc_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(hpc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(_vk_possible AND NOT ANDROID)
  target_link_libraries(hpc_core PRIVATE Vulkan::Headers)
endif()
if(ANDROID)
  target_link_libraries(hpc_core PRIVATE EGL GLESv3 android log)  # log: __android_log_print in log.cpp
endif()
if(MSVC)
  target_compile_options(hpc_core PRIVATE /W3 /utf-8)
  target_compile_definitions(hpc_core PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(hpc_core PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
endif()

if(HPC_BUILD_CLI)
  add_executable(hpc_cli cli/main.cpp)
  target_link_libraries(hpc_cli PRIVATE hpc_core)
  if(MSVC)
    target_compile_options(hpc_cli PRIVATE /utf-8)
    target_compile_definitions(hpc_cli PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
endif()

if(HPC_BUILD_TESTS)
  add_executable(hpc_tests tests/test_main.cpp)
  target_link_libraries(hpc_tests PRIVATE hpc_core)
  if(MSVC)
    target_compile_options(hpc_tests PRIVATE /utf-8)
    target_compile_definitions(hpc_tests PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
  add_test(NAME backends_match_fp64_reference COMMAND hpc_tests)
  if(_vk_possible)
    set(_hints)
    if(DEFINED ENV{VULKAN_SDK})
      list(APPEND _hints $ENV{VULKAN_SDK}/Bin $ENV{VULKAN_SDK}/bin)
    endif()
    find_program(HPC_SPIRV_VAL spirv-val HINTS ${_hints})
    if(HPC_SPIRV_VAL)
      foreach(spv ${HPC_SPV_FILES})  # spirv-val accepts one input file per invocation
        get_filename_component(spv_name ${spv} NAME_WE)
        add_test(NAME spirv_validates_${spv_name} COMMAND ${HPC_SPIRV_VAL} --target-env vulkan1.0 ${spv})
      endforeach()
    endif()
  endif()
endif()
