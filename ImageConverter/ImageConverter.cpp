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

cv::Mat ApplyAverageFilter(const cv::Mat& input)
{
    CV_Assert(input.type() == CV_8UC1);
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
                    sum += input.at<uchar>(y + ky, x + kx);
                }
            }
            output.at<uchar>(y, x) = static_cast<uchar>(sum / 9);
        }
    }
    return output;
}

cv::Mat ApplyAverageFilterWithColor(const cv::Mat& input)
{
	CV_Assert(input.type() == CV_8UC3);
	cv::Mat output = input.clone();

	for (int y = 1; y < input.rows - 1; y++)
	{
		for (int x = 1; x < input.cols - 1; x++)
		{
			cv::Vec3b sum = cv::Vec3b(0, 0, 0);
			for (int ky = -1; ky <= 1; ky++)
			{
				for (int kx = -1; kx <= 1; kx++)
				{
					cv::Vec3b pixel = input.at<cv::Vec3b>(y + ky, x + kx);
					sum += pixel;
				}
			}
			output.at<cv::Vec3b>(y, x) = sum / 9;
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

cv::Mat ApplySobelFilterWithColor(const cv::Mat& input, const std::vector<int>& kernel)
{
	CV_Assert(input.type() == CV_8UC3 && kernel.size() == 9);
	cv::Mat output = input.clone();

	for (int y = 1; y < input.rows - 1; y++)
	{
		for (int x = 1; x < input.cols - 1; x++)
		{
			cv::Vec3b sum = cv::Vec3b(0, 0, 0);
			for (int ky = -1; ky <= 1; ky++)
			{
				for (int kx = -1; kx <= 1; kx++)
				{
					cv::Vec3b pixel = input.at<cv::Vec3b>(y + ky, x + kx);
					sum += pixel * kernel[(ky + 1) * 3 + (kx + 1)];
				}
			}
			output.at<cv::Vec3b>(y, x) = sum;
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

        cv::Mat averagedImage = ApplyAverageFilter(grayImage);
        std::string averagedImagePath = baseName + "_filter1" + extension;
        cv::imwrite(averagedImagePath, averagedImage);

        std::vector<int> sobelKernel1 = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
        std::vector<int> sobelKernel2 = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
        
        cv::Mat sobelImage1 = ApplySobelFilter(grayImage, sobelKernel1);
        std::string sobelImagePath1 = baseName + "_filter2" + extension;
        cv::imwrite(sobelImagePath1, sobelImage1);

        cv::Mat sobelImage2 = ApplySobelFilter(grayImage, sobelKernel2);
        std::string sobelImagePath2 = baseName + "_filter3" + extension;
        cv::imwrite(sobelImagePath2, sobelImage2);

		cv::Mat averagedImageWithColor = ApplyAverageFilterWithColor(originalImage);
		std::string averagedImagePathWithColor = baseName + "_filter1_color" + extension;
		cv::imwrite(averagedImagePathWithColor, averagedImageWithColor);

		cv::Mat sobelImage1WithColor = ApplySobelFilterWithColor(originalImage, sobelKernel1);
		std::string sobelImagePath1WithColor = baseName + "_filter2_color" + extension;
		cv::imwrite(sobelImagePath1WithColor, sobelImage1WithColor);

		cv::Mat sobelImage2WithColor = ApplySobelFilterWithColor(originalImage, sobelKernel2);
		std::string sobelImagePath2WithColor = baseName + "_filter3_color" + extension;
		cv::imwrite(sobelImagePath2WithColor, sobelImage2WithColor);

        cv::imshow("Original Image", originalImage);
        cv::imshow("Grayscale Image (Filter 0)", grayImage);
        cv::imshow("Averaged Image (Filter 1)", averagedImage);
        cv::imshow("Sobel Filter 1 (Filter 2)", sobelImage1);
        cv::imshow("Sobel Filter 2 (Filter 3)", sobelImage2);
		cv::imshow("Averages Image with Color (Filter 1)", averagedImageWithColor);
		cv::imshow("Sobel Filter 1 With Color (Filter 2)", sobelImage1WithColor);
		cv::imshow("Sobel Filter 2 With Color (Filter 3)", sobelImage2WithColor);

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