#include <opencv2/opencv.hpp>
#include <queue>
#include <chrono>
#include <thread>
#include "frameQueue.h"
#include "cvxFont.h"
using namespace cvx;

LimitedQueue<cv::Mat> frameQueue(3);
std::mutex queueMutex;
bool stopThreads = false;

std::string fourccToString(int fourcc)
{
    char arr[5];
    arr[3] = (fourcc >> 24) & 0xFF;
    arr[2] = (fourcc >> 16) & 0xFF;
    arr[1] = (fourcc >> 8) & 0xFF;
    arr[0] = fourcc & 0xFF;
    arr[4] = '\0'; // Null terminator for C string

    printf("%d %d %d %d\n", arr[0], arr[1], arr[2], arr[3]);
    return std::string(arr);
}

void captureFrames()
{
    cv::VideoCapture cap(0, cv::CAP_V4L2);

    if (!cap.isOpened())
    {
        std::cerr << "Error opening video capture" << std::endl;
        return;
    }

    int fps = 30;
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    // cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
    // int width = 640;
    // int height = 480;
    int width = 1920;
    int height = 1080;
    cap.set(cv::CAP_PROP_FPS, fps);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    std::cout << "fps:" << cap.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "width:" << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "height:" << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "fourcc:" << fourccToString(cap.get(cv::CAP_PROP_FOURCC)) << std::endl;

    if (!cap.isOpened())
    {
        std::cerr << "Error opening camera" << std::endl;
        return;
    }

    int frame_count = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while (!stopThreads)
    {
        // auto start_in = std::chrono::high_resolution_clock::now();
        cv::Mat frame;
        if (cap.read(frame))
        {
            // std::cout << "push" << std::endl;

            frameQueue.push(frame);
            frame_count++;
        }
        // auto end_in = std::chrono::high_resolution_clock::now();
        // std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_in - start_in);
        //  std::cout << ms.count() << "ms\n";

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::seconds second = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        if (second.count() >= 20)
        {
            std::cout << "capture FPS:" << frame_count / second.count() << std::endl;
            break;
        }
    }
    cap.release();
}

void displayFrames()
{
    cv::namedWindow("Camera Feed", cv::WINDOW_KEEPRATIO);
    cv::setWindowProperty("Camera Feed", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    cvx::CvxFont font("/home/walter/rasp-camera/cpp/fonts/Noto.ttf");
    cv::String msg = "这段代码首先初始化FreeType库，然后加载宋体字体文件，并设置字体大小。";

    int frame_count = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while (!stopThreads)
    {

        if (!frameQueue.empty())
        {

            cv::Mat frame = frameQueue.front();
            frameQueue.pop();
            // auto start_in = std::chrono::high_resolution_clock::now();
            putText(frame, msg, cv::Point(100, 100), font, 30, cv::Scalar(0, 0, 255));
            // putText(frame, msg, cv::Point(700, 100), font, 30, cv::Scalar(255, 255, 255));
            // putText(frame, msg, cv::Point(100, 800), font, 30, cv::Scalar(255, 255, 255));
            // auto end_in = std::chrono::high_resolution_clock::now();
            // std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_in - start_in);
            // std::cout << ms.count() << "ms\n";
            cv::imshow("Camera Feed", frame);
            frame_count++;
            cv::waitKey(1);
        }
        else
        {
            // std::cout << "empty" << std::endl;
            cv::waitKey(1);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::seconds second = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        if (second.count() >= 20)
        {
            std::cout << "display FPS:" << frame_count / second.count() << std::endl;
            break;
        }
    }
}

int main()
{
    std::thread captureThread(captureFrames);
    std::thread displayThread(displayFrames);

    captureThread.join();
    displayThread.join();

    cv::destroyAllWindows();

    return 0;
}