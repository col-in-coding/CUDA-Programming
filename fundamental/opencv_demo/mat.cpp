#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

int main()
{
    std::string image_path = "/workspace/Github/CUDA-Programming/fundamental/Memory2D/starry_night.png";
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    // cv::imwrite("test.png", img);

    std::cout << "dims: " << img.dims << std::endl;
    std::cout << "rows: " << img.rows << std::endl;
    std::cout << "cols: " << img.cols << std::endl;

    std::cout << "channels(): " << img.channels() << std::endl;
    // 深度是表示每一个像素的位数，0表示CV_8U
    std::cout << "depth(): " << img.depth() << std::endl;
    std::cout << "isContinuous(): " << img.isContinuous() << std::endl;
    // type 16 表示 CV_8U channels为3
    std::cout << "type(): " << img.type() << std::endl;

    // 计算元素个数，不包含通道
    std::cout << "total: " << img.total() << std::endl;
    std::cout << "1 row: " << img.total(0, 1) << std::endl;
    // 与dims相对，每一维元素跨度，包含通道
    std::cout << "step0: " << img.step1(0) << std::endl;
    std::cout << "step1: " << img.step1(1) << std::endl;
    // 每个元素包含多少字节，elemSize1只算一个通道
    std::cout << "elemSize: " << img.elemSize() << std::endl;
    std::cout << "elemSize1: " << img.elemSize1() << std::endl; 

    // Rect(x, y, width, height)
    cv::Mat roi(img, cv::Rect(1, 1, 2, 2));
    // here passed by reference
    std::cout << "roi = " << std::endl << roi << std::endl;
    std::cout << "img ref cnt: " << img.u->refcount << std::endl;
    cv::Size size;
    cv::Point pt;
    roi.locateROI(size, pt);
    // 完整的size [width x height]
    std::cout << "size: " << size << std::endl; 
    std::cout << "point: " << pt << std::endl;

    // 随机初始化
    cv::Mat m = cv::Mat(1, 1, CV_8UC3);
    cv::randu(m, cv::Scalar::all(0), cv::Scalar::all(255));
    std::cout << "random matrix: " << std::endl << m << std::endl;
    // 数据访问
    int i = 0, j = 0;
    // at
    std::cout << "data at point: " << m.at<cv::Vec3b>(i, j) << std::endl;
    // Mat_
    cv::Mat_<cv::Vec3b> mat_ = m;
    std::cout << "data at point: " << mat_(i, j) << std::endl;
    // ptr
    cv::Vec3b* p = m.ptr<cv::Vec3b>(i);
    std::cout << "data at point: " << p[j] << std::endl;
    // data
    if (m.isContinuous()) {
        uchar* data = m.data;
        std::cout << "data at point: " << std::endl; 
        for (int k = 0; k < m.channels(); k++) {
            std::cout << +data[i * m.cols * m.channels() + j + k] << std::endl;
        }
    }

    return 0;
}