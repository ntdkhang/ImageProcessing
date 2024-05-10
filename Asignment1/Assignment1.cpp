#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

using namespace cv;
using namespace std;

void ReadImage(Mat& img) {
    string image_path = samples::findFile("./lena30.jpg");
    img = imread(image_path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "ERROR: Could not read the image: " << image_path << endl;
    }
} 



Mat BoxBlurFilter(Mat& original_img, int box_size) {
    /* 
     * This function takes an image, the size of the square box, and return a blurred image using the box blur
     */
    Size img_size = original_img.size();
    
    // TODO: Change blurred_img size to cut the border
    Mat blurred_img(img_size, CV_8UC1, Scalar(255));
    uchar* current_pixel;

    for (int row = box_size / 2; row < img_size.height - box_size / 2; ++row) {
        current_pixel = blurred_img.ptr<uchar>(row);
        for (int col = box_size / 2; col < img_size.width - box_size / 2; ++col) {
            int sum = 0;
            for (int i = row - box_size / 2; i <= row + box_size / 2; ++i) {
                for (int j = col - box_size / 2; j <= col + box_size / 2; ++j) {
                    sum += original_img.at<uchar>(i, j);
                }
            }
            current_pixel[col] = sum / (box_size * box_size);
        }
    }
    return blurred_img;

    /* Using convolution

    Size img_size = original_img.size();
    vector<vector<double>> kernel {{0.111,0.111,0.111}, {0.111,0.111,0.111}, {0.111,0.111,0.111}};
    Mat blurred_img(img_size, CV_8UC1, Scalar(0));
    original_img.copyTo(blurred_img);
    
    uchar* current_pixel;
    for (int row = 1; row < img_size.height - 1; ++row) {
        current_pixel = blurred_img.ptr<uchar>(row);
        for (int col = 1; col < img_size.width - 1; ++col) {
            // apply convolution
            double sum = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; ++j) {
                    sum += kernel[i][j] * original_img.at<uchar>(row + i - 1, col + j - 1);
                }
            }
            current_pixel[col] = sum;
        }
    }

    return blurred_img;
       */
}

vector<double> GetGaussianKernel15() {
    vector<double> kernel {
        0.00048872837522002, 
        0.009246250740395456,
        0.027839605612666265,
        0.06560233156931679,
        0.12099884565428047,
        0.1746973469158936,
        0.19744746769063704,
        0.1746973469158936,
        0.12099884565428047,
        0.06560233156931679,
        0.027839605612666265,
        0.009246250740395456,
        0.002403157286908872,
        0.00048872837522002
    };
    return kernel;
}

// https://www.cs.auckland.ac.nz/courses/compsci373s1c/PatricesLectures/Gaussian%20Filtering_1up.pdf
Mat GaussianFilter15x15(Mat& original_img) {
    Size img_size = original_img.size();
    vector<double> kernel { GetGaussianKernel15() };
    Mat blurred_img(img_size, CV_8UC1, Scalar(0));
    original_img.copyTo(blurred_img);
    
    // First convolution pass
    uchar* current_pixel;
    for (int row = 0; row < img_size.height; ++row) {
        current_pixel = blurred_img.ptr<uchar>(row);
        for (int col = 7; col < img_size.width - 7; ++col) {
            // apply convolution
            double sum = 0;
            for (int i = 0; i < 15; i++) {
                sum += kernel[i] * static_cast<int>(original_img.at<uchar>(row, col + i - 7));
            }
            current_pixel[col] = int(sum);
        }
    }

    // Second convolution pass
    Mat transposed_img;
    transpose(blurred_img, transposed_img);
    transpose(blurred_img, blurred_img);

    for (int row = 0; row < img_size.width; ++row) {
        current_pixel = transposed_img.ptr(row);
        for (int col = 7; col < img_size.height - 7; ++col) {
            // apply convolution
            double sum = 0;
            for (int i = 0; i < 15; i++) {
                sum += kernel[i] * blurred_img.at<uchar>(row, col + i - 7);
            }
            current_pixel[col] = int(sum);
        }
    }

    transpose(transposed_img, blurred_img);
    return blurred_img;
}
 
Mat MotionBlur(Mat& original_img, int size) {
    Size img_size = original_img.size();
    // vector<vector<int>> kernel {{0,-1,0}, {-1,4,-1}, {0,-1,0}};
    Mat blurred_img(img_size, CV_8UC1, Scalar(0));
    original_img.copyTo(blurred_img);
    
    // First convolution pass
    uchar* current_pixel;
    for (int row = size / 2; row < img_size.height - size / 2; ++row) {
        current_pixel = blurred_img.ptr<uchar>(row);
        for (int col = size / 2; col < img_size.width - size / 2; ++col) {
            // apply convolution with identity matrix
            int sum = 0;
            for (int i = -size / 2; i <= size / 2; i++) {
                sum += original_img.at<uchar>(row + i, col + i);
            }
            current_pixel[col] = sum / size;
        }
    }
    return blurred_img;
}


Mat LaplaceFilter(Mat& original_img) {
    Size img_size = original_img.size();
    vector<vector<int>> kernel {{0,-1,0}, {-1,4,-1}, {0,-1,0}};
    // vector<vector<double>> kernel {{0.111,0.111,0.111}, {0.111,0.111,0.111}, {0.111,0.111,0.111}};
    Mat blurred_img(img_size, CV_8UC1, Scalar(255));
    original_img.copyTo(blurred_img);
    
    // First convolution pass
    uchar* current_pixel;
    for (int row = 1; row < img_size.height - 1; ++row) {
        current_pixel = blurred_img.ptr<uchar>(row);
        for (int col = 1; col < img_size.width - 1; ++col) {
            // apply convolution
            int sum = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    sum += kernel[i + 1][j + 1] * original_img.at<uchar>(row + i, col + j);
                    if (row == 500) {
                        cout << (kernel[i + 1][j + 1]) << " ";
                    }
                }
            }
            if (row == 500)
                cout << sum << endl;
            sum = max(sum, 0);
            current_pixel[col] = sum;
        }
    }
    return blurred_img;
}

// Mat CannyEdgeDetection(Mat& original_img) {
//
// }


int main() {
    Mat img;
    ReadImage(img);

    // QUESTION 1
    Mat box_blurred = BoxBlurFilter(img, 7);
    imwrite("box_blurred.jpg", box_blurred);

    // QUESTION 2
    Mat gaussian_blurred = GaussianFilter15x15(img);
    imwrite("gaussian_blurred.jpg", gaussian_blurred);
    
    // QUESTION 3
    Mat motion_blurred = MotionBlur(img, 15);
    imwrite("motion_blurred.jpg", motion_blurred);

    // QUESTION 4
    Mat laplacian = LaplaceFilter(img);
    imwrite("laplacian.jpg", laplacian);
    // temp = MotionBlur(img, 15);

    // imshow("Original", img);
    // imshow("Laplacian", laplacian);
    // imshow("Blurred", gaussian_blurred);
    // waitKey(0);
    return 0;
}
