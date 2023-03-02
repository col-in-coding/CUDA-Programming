#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cmath>

// Standard CUDA API functions
#include <cuda_runtime_api.h>

// Error checking macro
#define cudaCheckError(code)                                             \
{                                                                        \
    if ((code) != cudaSuccess) {                                         \
      fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
              cudaGetErrorString(code));                                 \
    }                                                                    \
}

struct light {
  float x;
  float y;
  float radius;
  float brightness;
};

__device__ float light_brightness(
    float x, float y, size_t width, size_t height, const light &light)
{
    float norm_x = x / width;
    float norm_y = y / height;

    float dx = norm_x - light.x;
    float dy = norm_y - light.y;
    float distance_squared = dx * dx + dy * dy;
    if (distance_squared > light.radius * light.radius) {
        return 0;
    }
    float distance = sqrtf(distance_squared);
    float scaled_distance = distance / light.radius;
    if (scaled_distance > 0.8) {
        return (1.0f - (scaled_distance - 0.8f) * 5.0f) * light.brightness;
    } else {
        return light.brightness;
    }
}

template <typename T>
__device__ T *pointer2d(T *base_pointer, int x, int y, size_t pitch)
{
    return (T *)((uchar *)base_pointer + y * pitch) + x;
}

__device__ float clamp(float value) {return value > 1.0f ? 1.0f : value;}

__global__ void spotlights(
    uchar* input_data, uchar* output_data, size_t width, size_t height,
    size_t pitch, float ambient, light light1, light light2, light light3,
    light light4)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float brightness = ambient +
                       light_brightness(x, y, width, height, light1) +
                       light_brightness(x, y, width, height, light2) +
                       light_brightness(x, y, width, height, light3) +
                       light_brightness(x, y, width, height, light4);

    *pointer2d(output_data, x, y, pitch) =
        *pointer2d(input_data, x, y, pitch) * clamp(brightness);
}

int main()
{
    std::string image_path = "../data/starry_night.png";
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    cv::Mat output_img;
    output_img.create(img.rows, img.cols, CV_8UC3);
    light light1 = {0.2, 0.1, 0.1, 4.0};
    light light2 = {0.25, 0.2, 0.075, 2.0};
    light light3 = {0.5, 0.5, 0.3, 0.3};
    light light4 = {0.7, 0.65, 0.15, 0.8};

    size_t channels = img.channels();
    size_t width = img.cols * channels;
    size_t height = img.rows;
    size_t byte_width = width  * sizeof(uchar);
    size_t pitch;

    uchar *input_data_2d, *output_data_2d;
    // Allocate 2D aligned image
    cudaCheckError(
        cudaMallocPitch(&input_data_2d, &pitch, byte_width, height));
    cudaCheckError(
        cudaMallocPitch(&output_data_2d, &pitch, byte_width, height));
    cudaCheckError(
        cudaMemcpy2D(
            input_data_2d, pitch, img.data, byte_width,
            byte_width, height, cudaMemcpyHostToDevice));

    std::cout << "byte width: " << byte_width << std::endl;
    std::cout << "pitch: " << pitch << std::endl;

    dim3 block_dim(32, 16);
    dim3 grid_dim(
        (width + block_dim.x - 1) / block_dim.x,
        (height + block_dim.y - 1) / block_dim.y
    );

    spotlights<<<grid_dim, block_dim>>>(
        input_data_2d, output_data_2d, width, height, pitch, 0.3,
        light1, light2, light3, light4);

    cudaCheckError(
        cudaMemcpy2D(
            output_img.data, byte_width, output_data_2d, pitch,
            byte_width, height, cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(input_data_2d));
    cudaCheckError(cudaFree(output_data_2d));

    cv::imwrite("test.png", output_img);

    return 0;
}