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

/// @brief Provides drawing a rectangle, handles mouse events.
void processMouse(int, int, int, int, void*);

class demo_GrabCut : public IDemoCase {
public:
	demo_GrabCut();
	virtual std::string ReplyName() const override;
	friend void processMouse(int, int, int, int, void*);

private:
	/// @brief States of drawn rectangle
	enum ERectState {
		NOT_SET,		// there is no rectangle
		IN_PROCESS,		// the rectangle is being drawing now
		SET				// the rectangle is already drawn
	};

	static const int iterCount = 1;

	virtual void execute() override;
	void showOriginal();

	/// @brief Show OpenCV GrabCut
	void CVGrabCut();

	/// @brief Show OpenVX GrabCut
	void VXGrabCut();

	/// @brief Show difference between OpenCV and OpenVX
	void doDiff();

	/// @brief Initial point of rectangle angle
	cv::Point2i p;

	/// @brief Input rectangle around foreground
	cv::Rect m_rect;

	/// @brief Current rectangle drawing state
	ERectState m_rectState;

	/// @brief Image size
	cv::Size m_imgSize;

	/// @brief Input image to do GrabCut
	cv::Mat m_srcImage;

	/// @brief Temporary matrix for OpenCV GrabCut output image
	cv::Mat m_CVImage;

	/// @brief Temporary matrix for OpenVX GrabCut output image
	cv::Mat m_VXImage;

	/// @brief Temporary matrix for current difference between OpenCV and OpenVX
	cv::Mat m_diffImage;
};

demo_GrabCut::demo_GrabCut() {
	// empty
}

std::string demo_GrabCut::ReplyName() const {
	return "GrabCut segmentation";
}

namespace
{
	const std::string m_openVXWindow = "openVX";
	const std::string m_openCVWindow = "openCV";
	const std::string m_originalWindow = "original";
	const std::string m_diffWindow = m_openVXWindow + "-" + m_openCVWindow;
}

void demo_GrabCut::showOriginal() {
	cv::Mat result(m_imgSize, CV_8UC3);
	m_srcImage.copyTo(result);
	cv::rectangle(result, m_rect, cv::Scalar(0, 255, 0), 2);
	cv::imshow(m_originalWindow, result);
}

void demo_GrabCut::CVGrabCut() {
	m_CVImage = cv::Mat();
	cv::Mat cvBgdModel, cvFgdModel, cvMask;
	cvMask.create(m_imgSize, CV_8UC1);

	clock_t t1, t2;

	t1 = clock();

	cv::grabCut(m_srcImage, cvMask, m_rect, cvBgdModel, cvFgdModel, iterCount, cv::GC_INIT_WITH_RECT);

	t2 = clock();

	cv::Mat cvBinMask = cvMask & cv::GC_FGD;
	m_srcImage.copyTo(m_CVImage, cvBinMask);
	cv::imshow(m_openCVWindow, m_CVImage);
	char buf[32];
	sprintf_s(buf, " (time elapsed: %.3f secs)", (double)(t2 - t1) / CLOCKS_PER_SEC);
	cv::setWindowTitle(m_openCVWindow, m_openCVWindow + buf);
}

void demo_GrabCut::VXGrabCut() {
	uint32_t N = m_imgSize.width * m_imgSize.height;

	_vx_image srcVXImage = {
		m_srcImage.data,
		m_imgSize.width,
		m_imgSize.height,
		VX_DF_IMAGE_RGB,
		VX_COLOR_SPACE_DEFAULT
	};

	uint8_t* outVXImage = static_cast<uint8_t*>(calloc(N, sizeof(uint8_t)* 3));
	uint8_t* mask_data = static_cast<uint8_t*>(calloc(N, sizeof(uint8_t)));

	vx_rectangle_t vx_rect = {
		m_rect.x,
		m_rect.y,
		m_rect.x + m_rect.width,
		m_rect.y + m_rect.height
	};

	_vx_matrix mask = {
		mask_data,
		m_imgSize.height,
		m_imgSize.width,
		VX_TYPE_UINT8
	};

	clock_t t1, t2;

	t1 = clock();

	ref_GrabCutSegmentation(&srcVXImage, &mask, vx_rect, iterCount, VX_GC_INIT_WITH_RECT);

	t2 = clock();

	memset(outVXImage, 0, N * sizeof(uint8_t)* 3);
	for (vx_uint32 i = 0; i < N; i++) {
		if (mask_data[i] & VX_GC_FGD) {
			outVXImage[3 * i + 0] = m_srcImage.data[3 * i + 0];
			outVXImage[3 * i + 1] = m_srcImage.data[3 * i + 1];
			outVXImage[3 * i + 2] = m_srcImage.data[3 * i + 2];
		}
	}

	m_VXImage = cv::Mat(m_imgSize, CV_8UC3, outVXImage);
	cv::imshow(m_openVXWindow, m_VXImage);
	char buf[32];
	sprintf_s(buf, " (time elapsed: %.3f secs)", (double)(t2 - t1) / CLOCKS_PER_SEC);
	cv::setWindowTitle(m_openVXWindow, m_openVXWindow + buf);
}

void demo_GrabCut::doDiff() {
	m_diffImage.create(m_imgSize, CV_8UC3);
	cv::absdiff(m_VXImage, m_CVImage, m_diffImage);
	cv::imshow(m_diffWindow, m_diffImage);
}

void processMouse(int event, int x, int y, int, void *data) {
	demo_GrabCut *demo = static_cast<demo_GrabCut*>(data);
	switch (event) {
	case cv::EVENT_LBUTTONDOWN:
		if (demo->m_rectState != demo_GrabCut::IN_PROCESS) {
			demo->m_rectState = demo_GrabCut::IN_PROCESS;
			demo->p.x = x;
			demo->p.y = y;
			demo->m_rect.width = 5;
			demo->m_rect.height = 5;
			demo->showOriginal();
		}
		break;
	case cv::EVENT_LBUTTONUP:
		if (demo->m_rectState == demo_GrabCut::IN_PROCESS) {
			demo->m_rectState = demo_GrabCut::SET;
			demo->showOriginal();
			demo->CVGrabCut();
			demo->VXGrabCut();
			demo->doDiff();
		}
		break;
	case cv::EVENT_MOUSEMOVE:
		if (demo->m_rectState == demo_GrabCut::IN_PROCESS) {
			if (x < demo->p.x) {
				demo->m_rect.x = x;
				demo->m_rect.width = demo->p.x - x;
			}
			else if (x > demo->p.x) {
				demo->m_rect.x = demo->p.x;
				demo->m_rect.width = x - demo->p.x;
			}
			else {
				demo->m_rect.width = 5;
			}
			demo->m_rect.x = std::max(0, demo->m_rect.x);
			demo->m_rect.width = std::min(demo->m_rect.width, demo->m_imgSize.width - demo->m_rect.x);
			if (y < demo->p.y) {
				demo->m_rect.y = y;
				demo->m_rect.height = demo->p.y - y;
			}
			else if (y > demo->p.y) {
				demo->m_rect.y = demo->p.y;
				demo->m_rect.height = y - demo->p.y;
			}
			else {
				demo->m_rect.height = 5;
			}
			demo->m_rect.y = std::max(0, demo->m_rect.y);
			demo->m_rect.height = std::min(demo->m_rect.height, demo->m_imgSize.height - demo->m_rect.y);

			demo->showOriginal();
		}
		break;
	}
}

void demo_GrabCut::execute() {
	cv::namedWindow(m_originalWindow, CV_WINDOW_NORMAL);
	cv::namedWindow(m_openVXWindow, CV_WINDOW_NORMAL);
	cv::namedWindow(m_openCVWindow, CV_WINDOW_NORMAL);
	cv::namedWindow(m_diffWindow, CV_WINDOW_NORMAL);

	const std::string imgPath = "..\\Image\\apple.png";
	m_srcImage = cv::imread(imgPath, CV_LOAD_IMAGE_COLOR);
	m_imgSize = cv::Size(m_srcImage.cols, m_srcImage.rows);
	m_rectState = NOT_SET;
	cv::setMouseCallback(m_originalWindow, processMouse, static_cast<void*>(this));

	showOriginal();
	cv::waitKey(0);
}

IDemoCasePtr CreateGrabCutDemo()
{
	return std::make_unique<demo_GrabCut>();
}