#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolo/yolo.h"

int main()
{
    std::cout << cv::getVersionString() << std::endl;
    std::filesystem::path currentDir = std::filesystem::current_path();
    std::cout << "Current Directory: " << currentDir << std::endl;

    cv::Mat image(cv::imread("../people.jpeg"));

    std::string url = "http://34.45.120.21:8000";
    YoloClient yolo_client(url, "ensemble");

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<cv::Rect> boxes = yolo_client.detect(image);

    std::cout << "Inference time: " << util::duration(start_time) << std::endl;

    for (cv::Rect &box : boxes)
    {
        std::cout << box.x << " " << box.y << " " << box.height << " " << box.width << std::endl;
        cv::rectangle(image, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), cv::Scalar(255, 0, 0), 3);
    }

    cv::imwrite("output.jpeg", image);

    std::cout << "Total processing time: " << util::duration(start_time) << std::endl;

    return 0;
}
