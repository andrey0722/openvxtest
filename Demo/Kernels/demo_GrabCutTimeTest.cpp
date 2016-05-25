//@file demo_GrabCutSegmentation.cpp
//@brief Contains demonstration of GrabCutSegmentation function in comparing with OpenCV
//@author Andrey Olkhovsky
//@date 26 March 2016

#include "../stdafx.h"

#include <opencv2/opencv.hpp>

extern "C"
{
#include "Lib/Kernels/ref.h"
#include "Lib/Common/types.h"
}

#include "../DemoEngine.h"
#include <time.h>

class demo_GrabCutTimeTest : public IDemoCase {
public:
	demo_GrabCutTimeTest();
	virtual std::string ReplyName() const override;

private:

	virtual void execute() override;

	std::vector<std::string> imgPaths;
};

demo_GrabCutTimeTest::demo_GrabCutTimeTest() {
	// empty
}

std::string demo_GrabCutTimeTest::ReplyName() const {
	return "GrabCut segmentation time test";
}

void demo_GrabCutTimeTest::execute() {

	int choice = 0;
	std::cout << "Choose variant (VX = 0, CV = 1):" << std::endl;
	std::cin >> choice;
	std::cout << "--------------------------------------------------" << std::endl;

	const uint32_t sampleSize = 10;
	const int iterCount = 1;

	imgPaths.push_back("..\\Image\\apple16.png");
	imgPaths.push_back("..\\Image\\apple24.png");
	imgPaths.push_back("..\\Image\\apple32.png");
	imgPaths.push_back("..\\Image\\apple48.png");
	imgPaths.push_back("..\\Image\\apple64.png");
	imgPaths.push_back("..\\Image\\apple96.png");
	imgPaths.push_back("..\\Image\\apple128.png");
	imgPaths.push_back("..\\Image\\apple256.png");
	imgPaths.push_back("..\\Image\\apple320.png");
	imgPaths.push_back("..\\Image\\apple480.png");
	imgPaths.push_back("..\\Image\\apple512.png");
	imgPaths.push_back("..\\Image\\apple640.png");

	srand((unsigned)time(0));

	for (std::string imgPath : imgPaths) {
		cv::Mat srcImage = cv::imread(imgPath, CV_LOAD_IMAGE_COLOR);
		cv::Size imgSize = cv::Size(srcImage.cols, srcImage.rows);

		cv::Mat resultImage;

		std::cout << imgPath << std::endl;
		std::cout << imgSize.width * imgSize.height << " pixels" << std::endl;

		clock_t t1, t2;
		clock_t ticksSum = 0;

		for (uint32_t i = 0; i < sampleSize; i++) {
			uint32_t x = rand() % (imgSize.width / 3 - 1) + 1;
			uint32_t y = rand() % (imgSize.height * 4 / 7) + 1;
			uint32_t x_end = rand() % (imgSize.width / 3 - 1) + imgSize.width * 2 / 3;
			uint32_t y_end = imgSize.height - 1;
			cv::Rect rect(x, y, x_end - x, y_end - y);

			if (choice == 0) {
				uint32_t N = imgSize.width * imgSize.height;
				_vx_image srcVXImage = {
					srcImage.data,
					imgSize.width,
					imgSize.height,
					VX_DF_IMAGE_RGB,
					VX_COLOR_SPACE_DEFAULT
				};

				uint8_t* outVXImage = static_cast<uint8_t*>(calloc(N, sizeof(uint8_t)* 3));
				uint8_t* mask_data = static_cast<uint8_t*>(calloc(N, sizeof(uint8_t)));

				vx_rectangle_t vx_rect = {
					rect.x,
					rect.y,
					rect.x + rect.width,
					rect.y + rect.height
				};

				_vx_matrix mask = {
					mask_data,
					imgSize.height,
					imgSize.width,
					VX_TYPE_UINT8
				};

				t1 = clock();
				ref_GrabCutSegmentation(&srcVXImage, &mask, vx_rect, iterCount, VX_GC_INIT_WITH_RECT);
				t2 = clock();

				free(outVXImage);
				free(mask_data);
			}
			else {
				cv::Mat cvBgdModel, cvFgdModel, cvMask;
				cvMask.create(imgSize, CV_8UC1);

				t1 = clock();

				cv::grabCut(srcImage, cvMask, rect, cvBgdModel, cvFgdModel, iterCount, cv::GC_INIT_WITH_RECT);

				t2 = clock();
			}
			ticksSum += t2 - t1;
			double curTime = (double)(t2 - t1) / CLOCKS_PER_SEC;
			std::cout << "\t\t" << i << ": " << curTime << " secs" << std::endl;
		}

		std::cout << "\tMean: " << (double)ticksSum / sampleSize / CLOCKS_PER_SEC << std::endl;
	}
}

IDemoCasePtr CreateGrabCutTimeTestDemo()
{
	return std::make_unique<demo_GrabCutTimeTest>();
}