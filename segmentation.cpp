#include <algorithm>
#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include <opencv2/opencv.hpp>

#include "NvInferRuntimeCommon.h"

using std::filesystem::directory_iterator;

bool GetImageFiles(std::filesystem::path folderPath, std::vector<std::filesystem::path> &file_names)
{
    using namespace std::filesystem;
    directory_iterator iter(folderPath.string()), end;
    std::error_code err;

	std::vector<std::filesystem::path> extensions = {".jpg", ".png"};

    for (; iter != end && !err; iter.increment(err))
	 {
        const directory_entry entry = *iter;

		if (*find(extensions.begin(), extensions.end(), entry.path().extension()) == entry.path().extension())
		{
			file_names.push_back(entry.path());
		}
    }

    if (err)
	{
        std::cout << err.value() << std::endl;
        std::cout << err.message() << std::endl;
        return false;
    }

    return true;
}

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR))
        {
            std::cout << msg << "\n";
        }
    }
} gLogger;

struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            delete(obj);
        }
    }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
}

constexpr long long operator"" _MiB(long long unsigned val)
{
    return val * (1 << 20);
}

class Segmentation
{

public:
    Segmentation(std::filesystem::path engine_file_path, int32_t width, int32_t height, int class_num);
    ~Segmentation();
    void Run(std::vector<std::filesystem::path> input_images, std::filesystem::path output_directory);

private:
    std::string engine_file_path_;
    TRTUniquePtr<nvinfer1::ICudaEngine> engine_;
    TRTUniquePtr<nvinfer1::IExecutionContext> context_;
    size_t input_memory_size_;
    size_t output_memory_size_;
    int32_t width_;
    int32_t height_;
    int class_num_;

    bool Initializing();
    std::unique_ptr<float[]> PreProcess(cv::Mat* image);
    void PostProcess(cv::Mat* result_img, const int* buffer);
    std::unique_ptr<int> DoInference(const void* input_buffer);
    void CreateDisplayImage(cv::Mat* src, cv::Mat* mask, const double mask_alpha);
};

Segmentation::Segmentation(std::filesystem::path engine_file_path, int32_t width, int32_t height, int class_num)
    : engine_file_path_(engine_file_path.string())
    , engine_(nullptr)
    , input_memory_size_(0)
    , output_memory_size_(0)
    , width_(512)
    , height_(256)
    , class_num_(class_num)
{
    std::cout << "\033[32m[INFO ] Called c++ segmentation method. \033[m" << std::endl;
}

Segmentation::~Segmentation()
{
    std::cout << "\033[32m[INFO ] End c++ segmentation method \033[m" << std::endl;
}

bool Segmentation::Initializing()
{
    std::ifstream engine_file(engine_file_path_, std::ios::binary);
    if (engine_file.fail())
    {
        std::cout << "\033[31m[ERROR] engine_file.fail \033[m" << std::endl;
        return false;
    }

    engine_file.seekg(0, std::ifstream::end);
    auto file_size = engine_file.tellg();
    engine_file.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(file_size);
    engine_file.read(engineData.data(), file_size);

    TRTUniquePtr<nvinfer1::IRuntime> runtime{ nvinfer1::createInferRuntime(gLogger) };

    initLibNvInferPlugins(nullptr, "");
    engine_.reset(runtime->deserializeCudaEngine(engineData.data(), file_size, nullptr));

    assert(engine_.get() != nullptr);

    context_ = TRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_)
    {
        std::cout << "\033[31m[ERROR] !context \033[m" << std::endl;
        return false;
    }

    auto input_index = engine_->getBindingIndex("modelInput");
    if (input_index == -1)
    {
        std::cout << "\033[31m[ERROR] input_index == -1 \033[m" << std::endl;
        return false;
    }

    assert(engine_->getBindingDataType(input_index) == nvinfer1::DataType::kFLOAT);

    auto input_dimensions = nvinfer1::Dims4{ 1, 3, height_, width_ };
    context_->setBindingDimensions(input_index, input_dimensions);
    input_memory_size_ = getMemorySize(input_dimensions, sizeof(float));

    auto output_index = engine_->getBindingIndex("modelOutput");
    if (output_index == -1)
    {
        std::cout << "\033[31m[ERROR] output_index == -1 \033[m" << std::endl;
        return false;
    }

    assert(engine_->getBindingDataType(output_index) == nvinfer1::DataType::kINT32);

    auto output_dimensions = context_->getBindingDimensions(output_index);
    output_memory_size_ = getMemorySize(output_dimensions, sizeof(int32_t));

    return true;
}

std::unique_ptr<float[]> Segmentation::PreProcess(cv::Mat* image)
{
    const int height = height_;
    const int width = width_;
    const auto input_size = cv::Size(width, height);

    cv::Mat resized;
    cv::resize(*image, resized, input_size, 0, 0, cv::INTER_NEAREST);
    auto buffer = std::unique_ptr<float[]>{ new float[input_memory_size_] };

    constexpr double k = 1 / 255.;
    for (int i = 0; i < height * width; i++)
    {
        buffer.get()[i + 0 * height * width] = static_cast<float>(resized.at<cv::Vec3b>(i)[2] * k);
        buffer.get()[i + 1 * height * width] = static_cast<float>(resized.at<cv::Vec3b>(i)[1] * k);
        buffer.get()[i + 2 * height * width] = static_cast<float>(resized.at<cv::Vec3b>(i)[0] * k);
    }

    return buffer;
}

void Segmentation::PostProcess(cv::Mat* mask, const int* buffer)
{
    for (int i = 0; i < height_; i++)
    {
        for (int j = 0; j < width_; j++)
        {
            auto class_id = static_cast<uint8_t>(buffer[width_ * i + j]);
            if (class_id == 1)
            {
                mask->at<cv::Vec3b>(i, j) = { 0, 255, 0 };
            }

            else
            {
                mask->at<cv::Vec3b>(i, j) = { 0, 0, 0 };
            }
        }
    }
}

std::unique_ptr<int> Segmentation::DoInference(const void* buffer)
{
    void* input_cuda_memory = nullptr;
    if (cudaMalloc(&input_cuda_memory, input_memory_size_) != cudaSuccess)
    {
        std::cout << "\033[31m[ERROR] input cuda memory allocation failed \033[m" << std::endl;
        // gLogError << "ERROR: input cuda memory allocation failed, size = " << input_memory_size_ << " bytes" << std::endl;
    }

    void* output_cuda_memory = nullptr;
    if (cudaMalloc(&output_cuda_memory, output_memory_size_) != cudaSuccess)
    {
        std::cout << "\033[31m[ERROR] output cuda memory allocation failed \033[m" << std::endl;
        // gLogError << "ERROR: output cuda memory allocation failed, size = " << output_memory_size_ << " bytes" << std::endl;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        std::cout << "\033[31m[ERROR] CUDA stream creation failed \033[m" << std::endl;
        // gLogError << "ERROR: cuda stream creation failed." << std::endl;
    }

    // Copy image data to input binding memory
    if (cudaMemcpyAsync(input_cuda_memory, buffer, input_memory_size_, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        std::cout << "\033[31m[ERROR] CUDA memory copy of input failed \033[m" << std::endl;
        // gLogError << "ERROR: CUDA memory copy of input failed, size = " << input_memory_size_ << " bytes" << std::endl;
    }

    // Run TensorRT inference
    void* bindings[] = { input_cuda_memory, output_cuda_memory };
    if (!context_->enqueueV2(bindings, stream, nullptr))
    {
        std::cout << "\033[31m[ERROR] TensorRT inference failed \033[m" << std::endl;
        // gLogError << "ERROR: TensorRT inference failed" << std::endl;
    }

    // Copy predictions from output binding memory
    auto output_buffer = std::unique_ptr<int>{ new int[output_memory_size_] };
    if (cudaMemcpyAsync(output_buffer.get(), output_cuda_memory, output_memory_size_, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        std::cout << "\033[31m[ERROR] CUDA memory copy of output failed \033[m" << std::endl;
        // gLogError << "ERROR: CUDA memory copy of output failed, size = " << output_memory_size_ << " bytes" << std::endl;
    }

    cudaStreamSynchronize(stream);

    // Free CUDA resources
    cudaStreamDestroy(stream);
    cudaFree(input_cuda_memory);
    cudaFree(output_cuda_memory);

    return output_buffer;
}

void Segmentation::Run(std::vector<std::filesystem::path> input_images, std::filesystem::path output_directory)
{
    if (!Initializing())
    {
        // 
    };

    cv::Mat image;
    for (std::filesystem::path& file : input_images)
    {
        image = cv::imread(file.string());
        
        auto input_buffer = PreProcess(&image);
        auto output_buffer = DoInference(input_buffer.get());

        cv::Mat mask = cv::Mat::zeros(height_, width_, CV_8UC3);
        PostProcess(&mask, output_buffer.get());

        const auto input_size = cv::Size(image.cols, image.rows);

        cv::Mat resized;
        cv::resize(mask, resized, input_size, 0, 0, cv::INTER_NEAREST);

        // const double mask_alpha = 0.5;
        // CreateDisplayImage(&image, &resized, mask_alpha);

        auto output_patu = output_directory/file.filename();
        cv::imwrite(output_patu.string(), resized);
    }
}

void Segmentation::CreateDisplayImage(cv::Mat* image, cv::Mat* mask, const double mask_alpha)
{
    double bate = 1. - mask_alpha;

    for (int y = 0; y < image->rows; y++)
    {
        cv::Vec3b* ptr_image = image->ptr<cv::Vec3b>(y);
        cv::Vec3b* ptr_mask = mask->ptr<cv::Vec3b>(y);

        for (int x = 0; x < image->cols; x++)
        {
            cv::Vec3b value_image_bgr = ptr_image[x];
            cv::Vec3b value_mask_bgr = ptr_mask[x];
            cv::Vec3b new_value;

            int sum_pixel_value = 0;
            for (int i = 0; i < image->channels(); i++)
            {
                sum_pixel_value += value_mask_bgr[i];
            }

            for (int i = 0; i < image->channels(); i++)
            {
                if (sum_pixel_value < 60)
                {
                    new_value[i] = value_image_bgr[i];
                }
                else
                {
                    new_value[i] = (value_image_bgr[i] * bate) + (value_mask_bgr[i] * mask_alpha);
                }
            }

            ptr_image[x] = new_value;
        }
    }
}

int main(int argc, char** argv)
{
    int32_t width = 512;
    int32_t height = 256;
    int class_num = 2;

    std::filesystem::path engine = argv[1];
    std::filesystem::path target_directory = argv[2];
    std::filesystem::path output_directory = argv[3];
	std::vector<std::filesystem::path> file_names;

	auto status = GetImageFiles(target_directory, file_names);

    auto segmentation = std::make_shared<Segmentation>(engine, width, height, class_num);

    segmentation->Run(file_names, output_directory);

    return 0;
}