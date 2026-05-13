cmake_minimum_required(VERSION 3.28)
project(vector_based_fraud_detection_api)

set(CMAKE_CXX_STANDARD 26)

# -march=native: local dev; Dockerfile sobrescreve com -march=x86-64-v3 para portabilidade
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -funroll-loops")

# ──────────────────────────────────────────────
# PGO (Profile-Guided Optimization)
# Uso:
#   cmake -DPGO_GENERATE=ON ..   → gera instrumentação
#   cmake -DPGO_USE=ON ..        → consome perfil
# ──────────────────────────────────────────────
option(PGO_GENERATE "Instrument with -fprofile-generate" OFF)
option(PGO_USE      "Consume profile via -fprofile-use"  OFF)
set(PGO_DIR "${CMAKE_BINARY_DIR}/pgo-data" CACHE PATH "PGO profile data directory")

if(PGO_GENERATE AND PGO_USE)
    message(FATAL_ERROR "PGO_GENERATE and PGO_USE are mutually exclusive")
endif()

if(PGO_GENERATE)
    add_compile_options(-fprofile-generate=${PGO_DIR})
    add_link_options(-fprofile-generate=${PGO_DIR})
    message(STATUS "PGO: instrumentation mode (generate → ${PGO_DIR})")
elseif(PGO_USE)
    add_compile_options(-fprofile-use=${PGO_DIR} -fprofile-correction)
    add_link_options(-fprofile-use=${PGO_DIR} -fprofile-correction)
    message(STATUS "PGO: optimization mode (use ← ${PGO_DIR})")
endif()

# ──────────────────────────────────────────────
# LTO: controlado pela variável CPP_LTO.
# Para ativar localmente: cmake -DCPP_LTO=ON ..
# ──────────────────────────────────────────────
option(CPP_LTO "Enable Link-Time Optimization" OFF)
if(CPP_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_OUTPUT)
    if(IPO_SUPPORTED)
        message(STATUS "LTO/IPO: habilitado")
    else()
        message(STATUS "LTO/IPO: não suportado (${IPO_OUTPUT})")
    endif()
endif()

# ──────────────────────────────────────────────
# Hot path flags: sem exceções/RTTI, sem stack protector
# ──────────────────────────────────────────────
add_compile_options(
    -fno-stack-protector
    -fno-asynchronous-unwind-tables
    -ffast-math
    -fno-math-errno
)

find_package(Boost REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

# ──────────────────────────────────────────────
# simdjson: JSON parser SIMD-acelerado
# ──────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    simdjson
    GIT_REPOSITORY https://github.com/simdjson/simdjson.git
    GIT_TAG        v3.10.1
    GIT_SHALLOW    TRUE
)
set(SIMDJSON_BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
set(SIMDJSON_DEVELOPER_MODE OFF CACHE BOOL "" FORCE)
set(SIMDJSON_JUST_LIBRARY ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(simdjson)

include_directories(${Boost_INCLUDE_DIRS})

add_executable(vector_based_fraud_detection_api main.cpp)

if(CPP_LTO AND IPO_SUPPORTED)
    set_property(TARGET vector_based_fraud_detection_api
                 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

target_link_libraries(vector_based_fraud_detection_api PRIVATE
    Threads::Threads
    ZLIB::ZLIB
    simdjson::simdjson
)

if(NOT CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    add_custom_command(TARGET vector_based_fraud_detection_api POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_SOURCE_DIR}/resources
        ${CMAKE_CURRENT_BINARY_DIR}/resources
        COMMENT "Creating symbolic link for resources directory"
    )
endif()
