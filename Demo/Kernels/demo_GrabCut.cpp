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

class demo_GrabCut : public IDemoCase {
	virtual std::string ReplyName() const override;
	virtual void execute() override;
private:
	cv::Mat m_srcImage;
};

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

void demo_GrabCut::execute() {
	cv::namedWindow(m_originalWindow, CV_WINDOW_NORMAL);
	/*cv::namedWindow(m_openVXWindow, CV_WINDOW_NORMAL);
	cv::namedWindow(m_openCVWindow, CV_WINDOW_NORMAL);
	cv::namedWindow(m_diffWindow, CV_WINDOW_NORMAL);*/

	const std::string imgPath = "..\\Image\\apple.png";
	m_srcImage = cv::imread(imgPath, CV_LOAD_IMAGE_UNCHANGED);
	cv::imshow(m_originalWindow, m_srcImage);
	cv::waitKey(0);
}

IDemoCasePtr CreateGrabCutDemo()
{
	return std::make_unique<demo_GrabCut>();
}