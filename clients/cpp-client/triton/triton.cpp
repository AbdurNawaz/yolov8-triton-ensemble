#include <iostream>

#include <rapidjson/document.h>
#include "http_client.h"

#include "triton.h"

namespace tc = triton::client;

Triton::Triton(std::string tritonServerUrl, std::string modelName)
    : url(std::move(tritonServerUrl)), modelName(std::move(modelName))
{

    tc::Error err = tc::InferenceServerHttpClient::Create(&this->tritonClient.httpClient, this->url);

    if (!err.IsOk())
    {
        std::cerr << "Unable to create triton client: " << err << std::endl;
        exit(1);
    }
    // populate modelInfo
    this->getModelInfo();
}

void Triton::getModelInfo()
{
    /* This role of this function is to get the model metadata from yolo triton server. */

    ModelInfo modelInfo;

    std::string modelInfoJsonDump;
    tc::Error err = this->tritonClient.httpClient->ModelMetadata(&modelInfoJsonDump, this->modelName);
    if (!err.IsOk())
    {
        std::cerr << "Unable to get Model info: " << err << std::endl;
        exit(1);
    }

    rapidjson::Document modelInfoJson;
    modelInfoJson.Parse(modelInfoJsonDump.c_str());

    // parse input metadata
    modelInfo.inputName = modelInfoJson["inputs"][0]["name"].GetString();
    modelInfo.inputDtype = modelInfoJson["inputs"][0]["datatype"].GetString();
    for (const rapidjson::GenericValue<rapidjson::UTF8<>> &inputDim : modelInfoJson["inputs"][0]["shape"].GetArray())
    {
        modelInfo.inputDims.emplace_back(inputDim.GetInt64());
    }

    // parse output metadata
    for (const rapidjson::GenericValue<rapidjson::UTF8<>> &output : modelInfoJson["outputs"].GetArray())
    {
        modelInfo.outputNames.emplace_back(output["name"].GetString());
    }

    this->modelInfo = modelInfo;
}

std::vector<const tc::InferRequestedOutput *> Triton::createRequestedOutputs()
{
    std::vector<const tc::InferRequestedOutput *> outputs;

    for (const std::string &outputName : this->modelInfo.outputNames)
    {
        tc::InferRequestedOutput *output;
        tc::Error err = tc::InferRequestedOutput::Create(&output, outputName);
        outputs.push_back(output);
    }
    return outputs;
}

tc::InferResult *Triton::infer(const std::vector<uint8_t> &inputData)
{
    // create request inputs
    std::vector<tc::InferInput *> inputs{nullptr};
    tc::Error err = tc::InferInput::Create(
        &inputs.at(0), this->modelInfo.inputName, this->modelInfo.inputDims, this->modelInfo.inputDtype);
    if (!err.IsOk())
    {
        std::cerr << "Unable to create input placeholder: " << err << std::endl;
        exit(1);
    }
    err = inputs[0]->AppendRaw(inputData);
    if (!err.IsOk())
    {
        std::cerr << "Unable to append input data to placeholder: " << err << std::endl;
        exit(1);
    }

    // create requested outputs
    std::vector<const tc::InferRequestedOutput *> outputs = this->createRequestedOutputs();

    tc::InferOptions options(this->modelName);

    tc::InferResult *result;
    err = this->tritonClient.httpClient->Infer(&result, options, inputs, outputs);
    if (!err.IsOk() || !result->RequestStatus().IsOk())
    {
        std::cerr << "Unable to perform inference: " << err << std::endl;
        exit(1);
    }
    return result;
}

void Triton::parseFloatArrayFromResult(const tc::InferResult *result, const std::string &outputName, std::vector<float> &output)
{
    const uint8_t *outputData;
    size_t outputByteSize;
    tc::Error err = result->RawData(outputName, &outputData, &outputByteSize);
    if (!err.IsOk())
    {
        std::cerr << "Unable to get output for: " << outputName << ", " << err << std::endl;
        exit(1);
    }
    const auto floatOutputData = reinterpret_cast<const float *>(outputData);

    output.resize(outputByteSize / sizeof(float));
    std::memcpy(output.data(), floatOutputData, outputByteSize);
}
