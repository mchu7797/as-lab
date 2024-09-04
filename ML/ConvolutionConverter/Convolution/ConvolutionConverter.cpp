#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

cv::Mat ApplySobelFilter(const cv::Mat& input, const std::vector<int>& kernel)
{
    CV_Assert(input.type() == CV_8UC1 && kernel.size() == 9);
    cv::Mat output = cv::Mat::zeros(input.size(), CV_32F);

    for (int y = 1; y < input.rows - 1; y++)
    {
        for (int x = 1; x < input.cols - 1; x++)
        {
            float sum = 0;
            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    sum += input.at<uchar>(y + ky, x + kx) * kernel[(ky + 1) * 3 + (kx + 1)];
                }
            }
            output.at<float>(y, x) = sum;
        }
    }
    return output;
}

cv::Mat MaxPooling(const cv::Mat& input, int poolSize)
{
    CV_Assert(input.type() == CV_32F);
    int newRows = input.rows / poolSize;
    int newCols = input.cols / poolSize;
    cv::Mat output = cv::Mat::zeros(newRows, newCols, CV_32F);

    for (int y = 0; y < newRows; y++)
    {
        for (int x = 0; x < newCols; x++)
        {
            float maxVal = -std::numeric_limits<float>::max();
            for (int py = 0; py < poolSize; py++)
            {
                for (int px = 0; px < poolSize; px++)
                {
                    int inputY = y * poolSize + py;
                    int inputX = x * poolSize + px;
                    if (inputY < input.rows && inputX < input.cols)
                    {
                        maxVal = std::max(maxVal, input.at<float>(inputY, inputX));
                    }
                }
            }
            output.at<float>(y, x) = maxVal;
        }
    }
    return output;
}

void SaveMatrixToFile(const cv::Mat& matrix, const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int y = 0; y < matrix.rows; y++)
    {
        for (int x = 0; x < matrix.cols; x++)
        {
            file << matrix.at<float>(y, x) << " ";
            std::cout << matrix.at<float>(y, x) << " ";
        }
        file << std::endl;
        std::cout << std::endl;
    }
    file.close();
}

void ProcessImage(const std::string& imagePath)
{
    try
    {
        cv::Mat originalImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (originalImage.empty())
        {
            throw std::runtime_error("Failed to load image: " + imagePath);
        }

        std::cout << "Processing image: " << imagePath << std::endl;

        std::vector<int> sobelKernel = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };

        cv::Mat convOutput = ApplySobelFilter(originalImage, sobelKernel);

        std::cout << "Convolution output:" << std::endl;
        SaveMatrixToFile(convOutput, "convolution_output.txt");

        cv::Mat poolOutput = MaxPooling(convOutput, 2);

        std::cout << "\nPooling output:" << std::endl;
        SaveMatrixToFile(poolOutput, "pooling_output.txt");
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