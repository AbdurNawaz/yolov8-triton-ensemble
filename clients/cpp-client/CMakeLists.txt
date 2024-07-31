cmake_minimum_required(3.28)
project(cpp_client)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

add_executable(cpp_client
    main.cpp
    triton/triton.h
    triton/triton.cpp
    yolo/yolo.h
    yolo/yolo.cpp
)

set(TritonClient_DIR "/home/abdurnawaz_2011/clients/lib/cmake/TritonClient")
set(TritonCommon_DIR "/home/abdurnawaz_2011/clients/lib/cmake/TritonCommon")

find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)

target_include_directories(cpp_client PRIVATE "/home/abdurnawaz_2011/clients/include")
target_link_directories(cpp_client PRIVATE "/home/abdurnawaz_2011/clients/lib")

target_link_libraries(cpp_client PRIVATE httpclient grpcclient)

find_package(OpenCV REQUIRED)
target_link_libraries(cpp_client PRIVATE opencv_core opencv_highgui)

