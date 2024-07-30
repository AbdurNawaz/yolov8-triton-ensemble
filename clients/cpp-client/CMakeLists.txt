cmake_minimum_required(3.28)
project(cpp_client)

add_executable(cpp_client
    main.cpp
    triton/triton.h
    triton/triton.cpp
)

find_package(OpenCV REQUIRED)
