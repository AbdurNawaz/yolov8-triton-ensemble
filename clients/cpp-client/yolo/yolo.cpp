#include <chrono>
#include <opencv2/opencv.hpp>

#include "yolo.h"
#include "triton/trion.h"

YoloClient::YoloClient(std::string tritonUrl, std::string modelName)
{
    this->triton_ = std::make_unique<Triton>(std::move(tritonUrl), std::move(modelName));
}

void YoloClient::resize(cv::Mat &frame, int &newHeight, int &newWidth, int &padH, int &padW) const
{
    int srcHeight = frame.rows, srcWidth = frame.cols;
    newHeight = this->inputWidth, newWidth = this->inputHeight;
    if (this->keepRatio && srcHeight != srcWidth)
    {
        auto hwScale = static_cast<float>(srcHeight) / static_cast<float>(srcWidth);
        if (hwScale > 1)
        {
            newHeight = this->inputHeight;
            newWidth = static_cast<int>(this->inputWidth / hwScale);
            cv::resize(frame, frame, cv::Size(newWidth, newHeight), cv::INTER_AREA);
            padW = static_cast<int>((this->inputWidth - newWidth) * 0.5);
            cv::copyMakeBorder(frame, frame, 0, 0, padW, this->inputWidth - newWidth - padW, cv::BORDER_CONSTANT, 0);
        }
        else
        {
            newHeight = static_cast<int>(this->inputHeight * hwScale);
            newWidth = this->inputWidth;
            cv::resize(frame, frame, cv::Size(newWidth, newHeight), cv::INTER_AREA);
            padH = static_cast<int>((this->inputHeight - newHeight) * 0.5);
            cv::copyMakeBorder(frame, frame, padH, this->inputHeight - newHeight - padH, 0, 0, cv::BORDER_CONSTANT, 0);
        }
    }
    else
    {
        cv::resize(frame, frame, cv::Size(newWidth, newHeight), cv::INTER_AREA);
    }
}

std::tuple<int, int, float, float> YoloClient::preprocess(const cv::Mat &frame, std::vector<uint8_t> &inputData)
{

    cv::Mat frameCopy = frame.clone();

    cv::cvtColor(frameCopy, frameCopy, cv::COLOR_BGR2RGB);
    int newHeight = 0, newWidth = 0, padH = 0, padW = 0;

    this->resize(frameCopy, newHeight, newWidth, padH, padW);

    std::cout << "newHeight = " << newHeight << " newWidth = " << newWidth << std::endl;
    std::cout << "rows = " << frame.rows << " cols = " << frame.cols << std::endl;

    float scaleH = static_cast<float>(frame.rows) / static_cast<float>(newHeight);
    float scaleW = static_cast<float>(frame.cols) / static_cast<float>(newWidth);

    frameCopy.convertTo(frameCopy, CV_32FC3, 1.0f / 255.0f);

    // allocate the flattened array
    const size_t imageSize = frameCopy.total() * frameCopy.elemSize();
    inputData.resize(imageSize);

    std::cout << "frame.total() = " << frameCopy.total() << ", frame.elemSize() =  " << frameCopy.elemSize() << std::endl;

    std::vector<cv::Mat> imageRGBChannels;
    size_t pos = 0;
    for (int i = 0; i < 3; i++)
    {
        imageRGBChannels.emplace_back(
            this->inputHeight, this->inputWidth, CV_32FC1, &(inputData[pos]));
        pos += imageRGBChannels.back().total() * imageRGBChannels.back().elemSize();
    }

    cv::split(frameCopy, imageRGBChannels);

    if (pos != imageSize)
    {
        std::cerr << "unexpected total size of channels " << pos << ", expecting "
                  << imageSize << std::endl;
        exit(1);
    }

    return std::make_tuple(padH, padW, scaleH, scaleW);
}

std::vector<cv::Rect> YoloClient::detect(cv::Mat &frame)
{

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<uint8_t> inputData;
    const auto [padH, padW, scaleH, scaleW] = this->preprocess(frame, inputData);
    std::cout << "Split: " << duration(start_time) << std::endl;

    tc::InferResult *result = this->triton_->infer(inputData);

    auto [detectionBBoxes, detectionScores] = parseTritonOuput(result);

    std::cout << "Scores: " << std::endl;
    for (const float score : detectionScores)
    {
        std::cout << score << " ";
    }
    std::cout << std::endl
              << "Boxes: " << std::endl;
    ;
    for (const std::vector<float> &box : detectionBBoxes)
    {
        for (const float val : box)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::vector<cv::Rect> boxes = postProcess(detectionBBoxes, padH, padW, scaleH, scaleW);

    return boxes;
}

std::tuple<std::vector<std::vector<float>>, std::vector<float>> YoloClient::parseTritonOuput(tc::InferResult *tritonResult)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    // parse detection_bboxes
    std::vector<float> detectionBboxes;
    Triton::parseFloatArrayFromResult(tritonResult, "detection_bboxes", detectionBboxes);

    // resize to [-1, 4]
    std::vector<std::vector<float>> bboxes;
    bboxes.reserve(detectionBboxes.size() / 4);
    for (size_t i = 0; i < detectionBboxes.size(); i += 4)
    {
        bboxes.emplace_back(detectionBboxes.begin() + i, detectionBboxes.begin() + i + 4);
    }

    std::cout << "BBoxes size: " << bboxes.size() << std::endl;

    // parse detection_scores
    std::vector<float> detectionScores;
    Triton::parseFloatArrayFromResult(tritonResult, "detection_scores", detectionScores);

    std::cout << "parseTritonOutput: " << duration(start_time) << std::endl;

    return std::make_tuple(bboxes, detectionScores);
}

std::vector<cv::Rect> YoloClient::postProcess(std::vector<std::vector<float>> bboxes, int padH, int padW, float scaleH, float scaleW)
{
    /* This functoin scales the bboxes to the original frame size and converts (x1, y1, x2, y2) to (x, y, w, h) */

    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << padW << " " << padH << std::endl;
    std::cout << scaleW << " " << scaleH << std::endl;

    std::vector<float> padding = {static_cast<float>(padW), static_cast<float>(padH), static_cast<float>(padW), static_cast<float>(padH)};
    std::vector<float> scaling = {scaleW, scaleH, scaleW, scaleH};

    for (std::vector<float> &bbox : bboxes)
    {
        std::transform(bbox.begin(), bbox.end(), padding.begin(), bbox.begin(),
                       std::minus<>());
        std::transform(bbox.begin(), bbox.end(), scaling.begin(), bbox.begin(),
                       std::multiplies<>());
    }

    std::vector<cv::Rect> rectangles;
    for (const std::vector<float> &bbox : bboxes)
    {
        cv::Rect rectangle(
            static_cast<int>(bbox.at(0)),
            static_cast<int>(bbox.at(1)),
            static_cast<int>(bbox.at(2)) - static_cast<int>(bbox.at(0)),
            static_cast<int>(bbox.at(3)) - static_cast<int>(bbox.at(1)));
        rectangles.push_back(rectangle);
    }

    std::cout << "postProcess: " << duration(start_time) << std::endl;

    return rectangles;
}