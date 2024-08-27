#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

cv::Mat ConvertToGrayscale(const cv::Mat& input)
{
    CV_Assert(input.type() == CV_8UC3);
    cv::Mat output(input.rows, input.cols, CV_8UC1);
    for (int y = 0; y < input.rows; y++)
    {
        for (int x = 0; x < input.cols; x++)
        {
            cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);
            output.at<uchar>(y, x) = static_cast<uchar>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
        }
    }
    return output;
}

cv::Mat EqualizeHistogram(const cv::Mat& input)
{
    CV_Assert(input.type() == CV_8UC1);
    cv::Mat output = input.clone();
    int histogram[256] = { 0 };
    float cdf[256] = { 0.0f };

    for (int y = 0; y < input.rows; y++)
    {
        for (int x = 0; x < input.cols; x++)
        {
            histogram[input.at<uchar>(y, x)]++;
        }
    }

    int pixelCount = input.rows * input.cols;
    cdf[0] = static_cast<float>(histogram[0]) / pixelCount;
    for (int i = 1; i < 256; i++)
    {
        cdf[i] = cdf[i - 1] + static_cast<float>(histogram[i]) / pixelCount;
    }

    for (int y = 0; y < output.rows; y++)
    {
        for (int x = 0; x < output.cols; x++)
        {
            output.at<uchar>(y, x) = static_cast<uchar>(cdf[input.at<uchar>(y, x)] * 255);
        }
    }
    return output;
}

cv::Mat ApplySobelFilter(const cv::Mat& input, const std::vector<int>& kernel)
{
    CV_Assert(input.type() == CV_8UC1 && kernel.size() == 9);
    cv::Mat output = input.clone();

    for (int y = 1; y < input.rows - 1; y++)
    {
        for (int x = 1; x < input.cols - 1; x++)
        {
            int sum = 0;
            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    sum += input.at<uchar>(y + ky, x + kx) * kernel[(ky + 1) * 3 + (kx + 1)];
                }
            }
            output.at<uchar>(y, x) = cv::saturate_cast<uchar>(std::abs(sum));
        }
    }
    return output;
}

void ProcessImage(const std::string& imagePath)
{
    try
    {
        cv::Mat originalImage = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (originalImage.empty())
        {
            throw std::runtime_error("Failed to load image: " + imagePath);
        }

        std::cout << "Processing image: " << imagePath << std::endl;

        size_t lastDot = imagePath.find_last_of(".");
        size_t lastSlash = imagePath.find_last_of("/\\");
        std::string baseName = imagePath.substr(lastSlash + 1, lastDot - lastSlash - 1);
        std::string extension = imagePath.substr(lastDot);

        cv::Mat grayImage = ConvertToGrayscale(originalImage);
        std::string grayImagePath = baseName + "_filter0" + extension;
        cv::imwrite(grayImagePath, grayImage);

        cv::Mat equalizedImage = EqualizeHistogram(grayImage);
        std::string equalizedImagePath = baseName + "_filter1" + extension;
        cv::imwrite(equalizedImagePath, equalizedImage);

        std::vector<int> sobelKernel1 = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
        cv::Mat sobelImage1 = ApplySobelFilter(grayImage, sobelKernel1);
        std::string sobelImagePath1 = baseName + "_filter2" + extension;
        cv::imwrite(sobelImagePath1, sobelImage1);

        std::vector<int> sobelKernel2 = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
        cv::Mat sobelImage2 = ApplySobelFilter(grayImage, sobelKernel2);
        std::string sobelImagePath2 = baseName + "_filter3" + extension;
        cv::imwrite(sobelImagePath2, sobelImage2);

        cv::imshow("Original Image", originalImage);
        cv::imshow("Grayscale Image (Filter 0)", grayImage);
        cv::imshow("Equalized Image (Filter 1)", equalizedImage);
        cv::imshow("Sobel Filter 1 (Filter 2)", sobelImage1);
        cv::imshow("Sobel Filter 2 (Filter 3)", sobelImage2);

        cv::waitKey(0);
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "OpenCV exception: " << e.what() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown exception occurred" << std::endl;
    }
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <Image_Path>" << std::endl;
        return -1;
    }

    ProcessImage(argv[1]);
    return 0;
}