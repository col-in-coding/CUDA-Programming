#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

// Standard CUDA API functions
#include <cuda_runtime_api.h>

__global__ void monochrome(uchar* input_data, uchar* output_data, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;

    output_data[index] = input_data[index * 3] * 0.3125f +
                         input_data[index * 3 + 1] * 0.5f +
                         input_data[index * 3 + 2] * 0.1875f;
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
    output_img.create(img.rows, img.cols, CV_8UC1);

    /*
    // on CPU
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            cv::Vec3b inp = img.at<cv::Vec3b>(i, j);
            output_img.at<uchar>(i, j) = inp[0] * 0.1875f + inp[1] * 0.5f + inp[2] * 0.3125f;
        }
    }
    */

    // /*
    // on GPU
    {
        uchar *input_data_d, *output_data_d;
        // 这里不能使用 sizeof(img.data)
        int inp_bytes = img.total() * img.elemSize();
        int out_bytes = output_img.total() * output_img.elemSize();
        cudaMalloc(&input_data_d, inp_bytes);
        cudaMalloc(&output_data_d, out_bytes);
        
        if (img.isContinuous())
        {
            int pixel_count = img.rows * img.cols;
            int block_size = 256;
            int n_blocks = (pixel_count + block_size - 1) / block_size;
            cudaMemcpy(input_data_d, img.data, inp_bytes, cudaMemcpyHostToDevice);
            monochrome<<<n_blocks, block_size>>>(input_data_d, output_data_d, pixel_count);
            cudaMemcpy(output_img.data, output_data_d, out_bytes, cudaMemcpyDeviceToHost);
        }
    }

    cv::imwrite("test.png", output_img);
    // */

    return 0;
}