cmake_minimum_required(VERSION 3.28)
project(vector_based_fraud_detection_api)

set(CMAKE_CXX_STANDARD 26)

# -march=native: local dev; Dockerfile sobrescreve com -march=x86-64-v3 para portabilidade
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fno-rtti -funroll-loops")

# LTO: controlado pela variável de ambiente CPP_LTO (default: OFF).
# Para ativar localmente: cmake -DCPP_LTO=ON ..
# Não ativar via Dockerfile: GCC 13 lança ICE sob emulação QEMU x86-64.
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

find_package(Boost REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

add_executable(vector_based_fraud_detection_api main.cpp)

if(CPP_LTO AND IPO_SUPPORTED)
    set_property(TARGET vector_based_fraud_detection_api
                 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

target_link_libraries(vector_based_fraud_detection_api PRIVATE
    Threads::Threads
    ZLIB::ZLIB
)

if(NOT CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    add_custom_command(TARGET vector_based_fraud_detection_api POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_SOURCE_DIR}/resources
        ${CMAKE_CURRENT_BINARY_DIR}/resources
        COMMENT "Creating symbolic link for resources directory"
    )
endif()
