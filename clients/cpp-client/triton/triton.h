#pragma once

#include "http_client.h"

namespace tc = triton::client;

struct ModelInfo
{
    std::string inputName;
    std::string inputDtype;
    std::vector<int64_t> inputDims;
    std::vector<std::string> outputNames;
};

union TritonClient
{
    TritonClient()
    {
        new (&httpClient) std::unique_ptr<triton::client::InferenceServerHttpClient>{};
    }
    ~TritonClient() {}
    std::unique_ptr<triton::client::InferenceServerHttpClient> httpClient;
};

class Triton
{
public:
    Triton(std::string tritonServerUrl, std::string modelName);
    tc::InferResult *infer(const std::vector<uint8_t> &inputData);
    static void parseFloatArrayFromResult(const tc::InferResult *result, const std::string &outputName, std::vector<float> &output);

private:
    void getModelInfo();
    std::vector<const tc::InferRequestedOutput *> createRequestedOutputs();
    const std::string url;
    const std::string modelName;
    ModelInfo modelInfo;
    TritonClient tritonClient;
};
