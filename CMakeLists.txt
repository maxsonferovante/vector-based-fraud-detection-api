cmake_minimum_required(VERSION 3.28)
project(vector_based_fraud_detection_api)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Find Standard Packages
find_package(Boost REQUIRED COMPONENTS system)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenMP REQUIRED)

# Find OpenBLAS / LAPACK (standard in Linux/Docker)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED openblas)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# FAISS Configuration
# Try to find FAISS using find_package first
find_package(faiss QUIET)

if(NOT faiss_FOUND)
    message(STATUS "FAISS not found via find_package, using manual configuration")
    if(NOT FAISS_ROOT)
        set(FAISS_ROOT "/usr/local")
    endif()
    include_directories(${FAISS_ROOT}/include)
    link_directories(${FAISS_ROOT}/lib)
    set(FAISS_LIBRARIES faiss)
else()
    message(STATUS "FAISS found: ${faiss_CONFIG}")
    set(FAISS_LIBRARIES faiss)
endif()

include_directories(${OPENBLAS_INCLUDE_DIRS})
link_directories(${OPENBLAS_LIBRARY_DIRS})

add_executable(vector_based_fraud_detection_api main.cpp)

# Link everything
target_link_libraries(vector_based_fraud_detection_api PRIVATE 
    Boost::boost
    Threads::Threads
    ZLIB::ZLIB
    OpenMP::OpenMP_CXX
    ${FAISS_LIBRARIES}
    ${OPENBLAS_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
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
