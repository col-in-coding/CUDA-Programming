#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

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

    // 

    cv::imwrite("test.png", output_img);

    return 0;
}