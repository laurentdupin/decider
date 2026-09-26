cmake_minimum_required(VERSION 3.25)

project(InferBridgeDecider VERSION 0.1.0 LANGUAGES C CXX)

option(DECIDER_BUILD_HARNESS "Build the InferBridge llama.cpp harness" ON)
option(DECIDER_BUILD_TESTS "Build native Decider protocol tests" ON)
option(DECIDER_BUILD_CANARY "Build the model-backed InferBridge canary" ON)
option(DECIDER_LLAMA_VULKAN "Build llama.cpp with Vulkan support" OFF)

set(INFERBRIDGE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../third_party/Inferbridge"
    CACHE PATH "InferBridge source tree")
set(LLAMA_CPP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../third_party/llama.cpp"
    CACHE PATH "Pinned llama.cpp source tree")

if(NOT EXISTS "${INFERBRIDGE_ROOT}/include/inferbridge/inferbridge_harness.h")
    message(FATAL_ERROR "InferBridge harness headers were not found at ${INFERBRIDGE_ROOT}")
endif()
if(NOT EXISTS "${LLAMA_CPP_ROOT}/include/llama.h")
    message(FATAL_ERROR "llama.cpp was not found at ${LLAMA_CPP_ROOT}")
endif()

add_library(decider_native_protocol STATIC
    src/decider_protocol.cpp)
target_compile_features(decider_native_protocol PUBLIC cxx_std_17)
target_include_directories(decider_native_protocol
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/src"
    PRIVATE "${LLAMA_CPP_ROOT}/vendor")

if(MSVC)
    target_compile_options(decider_native_protocol PRIVATE /W4 /WX /permissive- /utf-8 /EHsc)
else()
    target_compile_options(decider_native_protocol PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(DECIDER_BUILD_HARNESS)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
    set(GGML_VULKAN ${DECIDER_LLAMA_VULKAN} CACHE BOOL "" FORCE)
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
    set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
    set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
    set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)

    add_subdirectory("${LLAMA_CPP_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

    add_library(inferbridge_decider_harness SHARED src/decider_harness.cpp)
    target_compile_features(inferbridge_decider_harness PRIVATE cxx_std_17)
    target_include_directories(inferbridge_decider_harness PRIVATE
        "${INFERBRIDGE_ROOT}/include"
        "${LLAMA_CPP_ROOT}/include"
        "${LLAMA_CPP_ROOT}/ggml/include"
        "${LLAMA_CPP_ROOT}/tools/mtmd"
        "${LLAMA_CPP_ROOT}/vendor")
    target_compile_definitions(inferbridge_decider_harness PRIVATE
        INFERBRIDGE_HARNESS_BUILDING_LIBRARY=1)
    target_link_libraries(inferbridge_decider_harness PRIVATE
        decider_native_protocol llama mtmd)
    set_target_properties(inferbridge_decider_harness PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        OUTPUT_NAME inferbridge_decider_harness)

    if(DECIDER_BUILD_CANARY)
        add_executable(decider_native_canary tools/harness_canary.cpp)
        target_compile_features(decider_native_canary PRIVATE cxx_std_17)
        target_include_directories(decider_native_canary PRIVATE
            "${INFERBRIDGE_ROOT}/include"
            "${LLAMA_CPP_ROOT}/vendor")
        target_link_libraries(decider_native_canary PRIVATE inferbridge_decider_harness)

        add_executable(decider_native_jsonl tools/harness_jsonl.cpp)
        target_compile_features(decider_native_jsonl PRIVATE cxx_std_17)
        target_include_directories(decider_native_jsonl PRIVATE
            "${INFERBRIDGE_ROOT}/include")
        target_link_libraries(decider_native_jsonl PRIVATE inferbridge_decider_harness)
    endif()
endif()

if(DECIDER_BUILD_TESTS)
    enable_testing()
    add_executable(decider_native_protocol_test tests/protocol_test.cpp)
    target_compile_features(decider_native_protocol_test PRIVATE cxx_std_17)
    target_include_directories(decider_native_protocol_test PRIVATE "${LLAMA_CPP_ROOT}/vendor")
    target_link_libraries(decider_native_protocol_test PRIVATE decider_native_protocol)
    add_test(NAME decider-native-protocol COMMAND decider_native_protocol_test)
    if(DECIDER_BUILD_HARNESS)
        add_executable(decider_native_abi_smoke tests/abi_smoke.cpp)
        target_compile_features(decider_native_abi_smoke PRIVATE cxx_std_17)
        target_include_directories(decider_native_abi_smoke PRIVATE "${INFERBRIDGE_ROOT}/include")
        target_link_libraries(decider_native_abi_smoke PRIVATE inferbridge_decider_harness)
        add_test(NAME decider-native-abi COMMAND decider_native_abi_smoke)
    endif()
endif()
