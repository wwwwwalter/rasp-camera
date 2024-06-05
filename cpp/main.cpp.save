#include <opencv2/opencv.hpp>  
#include <iostream>  
#include "decode.h"

std::string fourccToString(int fourcc) {
    char arr[5];
    arr[3] = (fourcc >> 24) & 0xFF;
    arr[2] = (fourcc >> 16) & 0xFF;
    arr[1] = (fourcc >> 8) & 0xFF;
    arr[0] = fourcc & 0xFF;
    arr[4] = '\0'; // Null terminator for C string  

    printf("%d %d %d %d\n", arr[0], arr[1], arr[2], arr[3]);
    return std::string(arr);
}

int main() {
    //test_decode();
    //return 0;

    /*  定义VideoCapture  */
    // 打开相机，0 表示默认的相机  
    cv::VideoCapture cap(0, cv::CAP_V4L2);//windows

    if (!cap.isOpened()) {
        std::cerr << "无法打开相机" << std::endl;
        return -1;
    }

    int fps = 30;
    //int width = 640;
    //int height = 480;

    int width = 1920;
    int height = 1080;
    cap.set(cv::CAP_PROP_FPS, fps);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    //cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));


    std::cout << "fps:" << cap.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "width:" << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "height:" << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "fourcc:" << fourccToString(cap.get(cv::CAP_PROP_FOURCC)) << std::endl;

    /*  定义VideoWriter  */
    // 定义视频编解码器并设置属性  
    //int fourcc = cv::VideoWriter::fourcc('M', 'P', '4', 'V'); // 选择MP4V编解码器  
    int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1'); // 选择H264编解码器

    //int fps = 30; // 设置视频帧率  
    //cv::Size frameSize(640, 480); // 设置视频帧大小，可以根据需要调整  
    // 获取相机帧的原始大小  
    cv::Size frameSize = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),(int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << frameSize << std::endl;

    // 创建VideoWriter对象，准备写入视频文件  
    cv::VideoWriter writer;
    writer.open("output.mp4", fourcc, fps, frameSize, true);

    if (!writer.isOpened()) {
        std::cerr << "无法打开视频文件以写入" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        // 从相机捕获一帧图像  
        if (!cap.read(frame)) {
            std::cerr << "无法从相机读取帧" << std::endl;
            break;
        }



        // 将帧写入视频文件  
        writer.write(frame);

        // 显示帧（可选，主要用于调试）  
        cv::imshow("Camera Feed", frame);

        // 等待按键或延迟一段时间，以便观察视频流（可选）  
        if (cv::waitKey(1000 / fps) >= 0) {
            break;
        }
    }

    // 释放资源并关闭窗口  
    cap.release();
    writer.release();
    cv::destroyAllWindows();

    return 0;
}