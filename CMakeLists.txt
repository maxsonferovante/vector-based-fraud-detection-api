cmake_minimum_required(VERSION 3.28)
project(vector_based_fraud_detection_api)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Find Standard Packages
find_package(Boost REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

add_executable(vector_based_fraud_detection_api main.cpp)

# Link everything
target_link_libraries(vector_based_fraud_detection_api PRIVATE 
    Threads::Threads
    ZLIB::ZLIB
)

# Symlink resources for local development
if(NOT CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    add_custom_command(TARGET vector_based_fraud_detection_api POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink 
        ${CMAKE_SOURCE_DIR}/resources 
        ${CMAKE_CURRENT_BINARY_DIR}/resources
        COMMENT "Creating symbolic link for resources directory"
    )
endif()
