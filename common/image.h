#ifndef __IMAGE__
#define __IMAGE__

#include <iostream>
using namespace std;

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

class IMAGE
{
public:
    Mat image;


    IMAGE( int w, int h) 
    {
        image = Mat::zeros(w,h,CV_8UC4);
        imshow("images",image);
    }


    unsigned char* get_ptr( void ) const   
    { 
        return (unsigned char*)image.data; 
    }

    long image_size( void ) const 
    { 
		return image.cols * image.rows * 4; 
    }


    char show_image(int time=0)
    {
        cv::namedWindow("camera", CV_WINDOW_NORMAL);//CV_WINDOW_NORMAL就是0
        cv::resizeWindow("camera", 1920, 1080);
        cv::imshow("camera", image);
        return waitKey(time);
    }

};



#endif  // __IMAGE__

