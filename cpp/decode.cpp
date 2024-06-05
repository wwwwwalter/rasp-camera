#include <opencv2/opencv.hpp>  
#include <iostream>  

int test_decode() {
    // 设定视频参数  
    int fps = 30;
    cv::Size frameSize(640, 480);

    // 尝试使用'MP4V'编解码器  
    cv::VideoWriter writer_mp4v;
    writer_mp4v.open("test_mp4v.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frameSize, true);
    if (writer_mp4v.isOpened()) {
        std::cout << "MP4V codec is supported." << std::endl;
        writer_mp4v.release(); // 关闭文件并释放资源  
    }
    else {
        std::cout << "MP4V codec is not supported." << std::endl;
    }

    // 尝试使用'H264'编解码器  
    cv::VideoWriter writer_h264;
    writer_h264.open("test_h264.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, frameSize, true);
    if (writer_h264.isOpened()) {
        std::cout << "H264 codec is supported." << std::endl;
        writer_h264.release(); // 关闭文件并释放资源  
    }
    else {
        std::cout << "H264 codec is not supported." << std::endl;
    }

    return 0;
}