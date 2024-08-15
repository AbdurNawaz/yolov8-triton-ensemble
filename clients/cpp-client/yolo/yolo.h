#pragma once

#include <chrono>
#include <opencv2/opencv.hpp>
#include "http_client.h"

#include "triton/triton.h"

namespace tc = triton::client;

namespace util
{
    inline double duration(std::chrono::high_resolution_clock::time_point time)
    {
        std::chrono::high_resolution_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(currentTime - time).count();
    }
}

class YoloClient
{
public:
    YoloClient(std::string tritonUrl, std::string modelName);
    std::vector<cv::Rect> detect(cv::Mat &frame);

private:
    const int inputHeight = 640;
    const int inputWidth = 640;
    bool keepRatio = true;
    const int numClasses = 1;
    const int regMax = 16;
    const float confThreshold = 0.45;
    const float nmsThreshold = 0.5;
    std::unique_ptr<Triton> triton_;

    void resize(cv::Mat &frame, int &newHeight, int &newWidth, int &padH, int &padW) const;
    std::tuple<int, int, float, float> preprocess(const cv::Mat &frame, std::vector<uint8_t> &inputData);
    static std::vector<cv::Rect> postProcess(std::vector<std::vector<float>> bboxes, int padH, int padW, float scaleH, float scaleW);
    static std::tuple<std::vector<std::vector<float>>, std::vector<float>> parseTritonOuput(tc::InferResult *tritonResult);
    const std::string url;
};
