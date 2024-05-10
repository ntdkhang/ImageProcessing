#include <iostream>
#include <opencv2/core/base.hpp>
#include <opencv2/core/fast_math.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using std::cout;
using std::endl;
using std::vector;
using namespace cv;


void part_a(Mat& img_color) {
    // Grey histogram
    // https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
    Mat img;
    cvtColor(img_color, img, COLOR_BGR2GRAY);
    Mat hist;
    int hist_size = 256;
    float range[] = {0, 256};
    const float* hist_range[] = {range};
    calcHist(&img, 1, 0, Mat(), hist, 1, &hist_size, hist_range);


    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double) hist_w / hist_size);
    Mat hist_img = Mat::zeros(hist_h, hist_w, CV_8UC1);
    normalize(hist, hist, 0, hist_img.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hist_size; i++) {
        int bin_val = cvRound(hist.at<float>(i - 1));
        int next_bin_val = cvRound(hist.at<float>(i));
        line(hist_img, Point(bin_w*(i-1), hist_h - bin_val), 
                Point(bin_w*(i), hist_h - next_bin_val),
                Scalar(255, 0 , 0), 2, 8, 0);
    }
    // imshow("Histogram", hist_img);
    imwrite("./result/histogram.jpg", hist_img);
    // waitKey(0);

    // Binary threshold
    Mat thres_binary, thres_four;
    threshold(img, thres_binary, 127, 255, THRESH_BINARY);
    // imshow("Threshold Binary", thres_binary);
    imwrite("./result/thres_binary.jpg", thres_binary);

    // three threshold values 
    Mat look_up_table(1, 256, CV_8U);
    uchar* p = look_up_table.ptr();
    for (int i = 0; i < 256; i++) {
        if (i <= 43) { p[i] = 0; }
        else if (i <= 128) { p[i] = 85; }
        else if (i <= 213) { p[i] = 170; }
        else { p[i] = 255; }
    }
    LUT(img, look_up_table, thres_four);
    // imshow("Threshold four", thres_four);
    imwrite("./result/thres_four.jpg", thres_four);
    // waitKey(0);

}

void part_b(Mat& img) {
    Mat data = Mat::zeros(img.cols*img.rows, 3, CV_32F);
    Mat best_labels, centers, clustered;
    std::vector<Mat> bgr;
    split(img, bgr);

    for(int i=0; i<img.cols*img.rows; i++) {
        data.at<float>(i,0) = bgr[0].data[i] / 255.0;
        data.at<float>(i,1) = bgr[1].data[i] / 255.0;
        data.at<float>(i,2) = bgr[2].data[i] / 255.0;
    }

    int num_cluster = 6;
    kmeans(data, num_cluster, best_labels,
            TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
            5, KMEANS_PP_CENTERS, centers);


    centers = centers.reshape(3, centers.rows);
    data = data.reshape(3, data.rows);
    clustered = Mat(img.rows, img.cols, CV_32F);

    Vec3f* p = data.ptr<Vec3f>();

    for (int i = 0; i < data.rows; i++) {
        int id = best_labels.at<int>(i);
        p[i] = centers.at<Vec3f>(id);
    }
    clustered = data.reshape(3, img.rows);
    // imshow("original", img);
    // imshow("clustered", clustered);
    imwrite("./result/clustered.jpeg", 255 * clustered); // clustered is stored in floating point values in range [0...1], so need to scale it in order to write    

    // waitKey();
}

void part_c(Mat& img) {
    Mat canny_output;
    Mat img_gray;
    int thresh = 80;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    Canny(img_gray, canny_output, thresh, thresh * 2);
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_TC89_KCOS);

    Mat drawing = img.clone(); // Mat::zeros(canny_output.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(0, 0, 255);
        drawContours(drawing, contours, i, color, 2, LINE_8, hierarchy, 0);
    }
    // imshow("Contours", drawing);
    imwrite("./result/contours.jpg", drawing);

    // Canny()
}

int main() {
    // Read img
    Mat img;
    img = imread("./sample.jpeg", IMREAD_COLOR);
    part_a(img);
    part_b(img);
    part_c(img);
    return 0;
}
