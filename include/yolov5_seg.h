//
// Created by simmons on 8/10/23.
//

#ifndef EDGEDIRECTVO_YOLOV5_SEG_H
#define EDGEDIRECTVO_YOLOV5_SEG_H
#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"
#include"cuda.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <string>
#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            sample::gLogError << "Cuda failure: " << ret << std::endl;                                                 \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#include "cuda_runtime_api.h"
using namespace nvinfer1;
using namespace plugin;
static Logger gLogger;
const static int kOutputSize1 = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
const static int kOutputSize2 = 32 * (kInputH / 4) * (kInputW / 4);

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw, std::string& img_dir, std::string& labels_filename) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        std::cout<<"wts ==> "<<wts<<"engine ===> "<<engine<<std::endl;
        auto net = std::string(argv[4]);
        if (net[0] == 'n') {
            gd = 0.33;
            gw = 0.25;
        } else if (net[0] == 's') {
            gd = 0.33;
            gw = 0.50;
        } else if (net[0] == 'm') {
            gd = 0.67;
            gw = 0.75;
        } else if (net[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
        } else if (net[0] == 'x') {
            gd = 1.33;
            gw = 1.25;
        } else if (net[0] == 'c' && argc == 7) {
            gd = atof(argv[5]);
            gw = atof(argv[6]);
        } else {
            return false;
        }
    } else if (std::string(argv[1]) == "-d" && argc == 5) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
        labels_filename = std::string(argv[4]);
    } else {
        return false;
    }
    return true;
}

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer1, float** gpu_output_buffer2, float** cpu_output_buffer1, float** cpu_output_buffer2) {
    assert(engine->getNbBindings() == 3);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex1 = engine->getBindingIndex(kOutputTensorName);
    const int outputIndex2 = engine->getBindingIndex("proto");
    assert(inputIndex == 0);
    assert(outputIndex1 == 1);
    assert(outputIndex2 == 2);

    // Create GPU buffers on device
//    CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
//    CHECK(cudaMalloc((void**)gpu_output_buffer1, kBatchSize * kOutputSize1 * sizeof(float)));
//    CHECK(cudaMalloc((void**)gpu_output_buffer2, kBatchSize * kOutputSize2 * sizeof(float)));
    cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float));
    cudaMalloc((void**)gpu_output_buffer1, kBatchSize * kOutputSize1 * sizeof(float));
    cudaMalloc((void**)gpu_output_buffer2, kBatchSize * kOutputSize2 * sizeof(float));

    // Alloc CPU buffers
    *cpu_output_buffer1 = new float[kBatchSize * kOutputSize1];
    *cpu_output_buffer2 = new float[kBatchSize * kOutputSize2];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output1, float* output2, int batchSize) {
    context.enqueue(batchSize, buffers, stream, nullptr);
//    CUDA_CHECK(cudaMemcpyAsync(output1, buffers[1], batchSize * kOutputSize1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
//    CUDA_CHECK(cudaMemcpyAsync(output2, buffers[2], batchSize * kOutputSize2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaMemcpyAsync(output1, buffers[1], batchSize * kOutputSize1 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(output2, buffers[2], batchSize * kOutputSize2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

void serialize_engine(unsigned int max_batchsize, float& gd, float& gw, std::string& wts_name, std::string& engine_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = nullptr;

    engine = build_seg_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);

    assert(engine != nullptr);

    // Serialize the engine
    IHostMemory* serialized_engine = engine->serialize();
    assert(serialized_engine != nullptr);

    // Save engine to file
    std::cout<<"engine_name ===> "<<engine_name<<std::endl;
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cerr << "Could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
    serialized_engine->destroy();
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();


    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

cv::Mat predict(const cv::Mat &img){
    cudaSetDevice(kGpuId);

    std::string wts_name = "";
    std::string engine_name = "/home/rushmian/tensorrtx/yolov5/build/yolov5n-seg.engine";
    std::string labels_filename = "/home/rushmian/tensorrtx/yolov5/build/coco.txt";
    float gd = 0.0f, gw = 0.0f;
    std::string img_dir;

    // Deserialize the engine from file
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
//    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaStreamCreate(&stream);

    // Init CUDA preprocessing
    cuda_preprocess_init(kMaxInputImageSize);

    // Prepare cpu and gpu buffers
    float* gpu_buffers[3];
    float* cpu_output_buffer1 = nullptr;
    float* cpu_output_buffer2 = nullptr;
    prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &gpu_buffers[2], &cpu_output_buffer1, &cpu_output_buffer2);

    // Read the txt file for classnames
    std::ifstream labels_file(labels_filename, std::ios::binary);
    if (!labels_file.good()) {
        std::cerr << "read " << labels_filename << " error!" << std::endl;
        exit(-1);
    }
    std::unordered_map<int, std::string> labels_map;
    read_labels(labels_filename, labels_map);
    assert(kNumClass == labels_map.size());

    // batch predict
    std::vector<cv::Mat> img_batch;
    std::vector<std::string> img_name_batch;
    img_batch.push_back(img);
//    img_name_batch.push_back(file_names[j]);

    // Preprocess
    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // Run inference
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer1, cpu_output_buffer2, kBatchSize);
    auto end = std::chrono::system_clock::now();
//    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // NMS
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, cpu_output_buffer1, img_batch.size(), kOutputSize1, kConfThresh, kNmsThresh);

    // Draw result and save image
    auto& res = res_batch[0];
    auto masks = process_mask(&cpu_output_buffer2[0 * kOutputSize2], kOutputSize2, res);
    cv::Mat img_copy = img.clone();
    cv::Mat outputMask;
    cv::Mat _;
    std::cout<<masks.size()<<std::endl;
    if(masks.size()!=0) {
        draw_mask_bbox(img_copy, res, masks, labels_map);
        cv::imwrite("result.png", img_copy);
        outputMask = scale_mask(masks[0], img_copy);
        size_t elements = masks.size();
        for(int i=1; i<elements; ++i){
            outputMask += scale_mask(masks[i], img_copy);
        }
//        std::cout<<outputMask<<std::endl;
        cv::threshold(outputMask, outputMask, 0.0, 255.0, cv::THRESH_BINARY);
//        outputMask.convertTo(outputMask, CV_8U);
//        cv::imshow("mask", outputMask);
//        std::cout<<"type: "<<outputMask.type()<<std::endl;
//        cv::waitKey(-1);
//        cv::destroyAllWindows();
    }
    assert(outputMask.size == img_copy.size);

    // Release stream and buffers
    cudaStreamDestroy(stream);
//    CUDA_CHECK(cudaFree(gpu_buffers[0]));
//    CUDA_CHECK(cudaFree(gpu_buffers[1]));
//    CUDA_CHECK(cudaFree(gpu_buffers[2]));
    cudaFree(gpu_buffers[0]);
    cudaFree(gpu_buffers[1]);
    cudaFree(gpu_buffers[2]);
    delete[] cpu_output_buffer1;
    delete[] cpu_output_buffer2;
    cuda_preprocess_destroy();
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return outputMask;
}
#endif //EDGEDIRECTVO_YOLOV5_SEG_H
