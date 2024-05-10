#include <iostream>
#include <filesystem>
#include <opencv2/core/saturate.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>


using namespace cv;
using namespace cv::ml;
using std::cout;
using std::endl;
using std::vector;
using std:: string;

void adjust_brightness(Mat& img) {
    double avg_brightness = cv::mean(img)[0] / 255; // cv::mean returns a Scalar (4-element vector). In this case, only the first element have a value, 
                                                    // all other 3 elements is 0. So get first element 
                                                    // adjust brightness
    if (avg_brightness > 0.6 || avg_brightness < 0.4) {
        // decrease brightness
        double factor = 0.5 / avg_brightness;
        Mat new_img(img.size(), CV_8UC1, Scalar(0));
        Mat look_up_table(1, 256, CV_8U);
        uchar* p = look_up_table.ptr();
        for (int i = 0; i < 256; i++) {
            p[i] = saturate_cast<uchar>(i * factor);
        }
        LUT(img, look_up_table, new_img);
        img = new_img;
    }
}

Mat question1(Mat& img, std::string img_name, bool write_file = false) {
    adjust_brightness(img);
    // resize
    Mat img_200, img_50;
    resize(img, img_200, Size(200, 200));
    resize(img, img_50, Size(50, 50));
    if (write_file) {
        std::string dir = "./ResultImgs/" + img_name;
        imwrite(dir + "_200.jpg", img_200);
        imwrite(dir + "_50.jpg", img_50);
    }
    img_50.convertTo(img_50, CV_32F);
    img_50 = img_50.reshape(1, 1);
    return img_50;
}

Mat question2(const Mat& img, std::string img_name, bool write_file = false) {
    cv::Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
    Ptr<SiftDescriptorExtractor> descriptor = SiftDescriptorExtractor::create();
    std::vector<KeyPoint> keypoints;
    Mat sift_feature_img;
    Mat sift_features(2048, 128, CV_32F);

    detector->detect(img, keypoints);
    descriptor->compute(img, keypoints, sift_features);

    drawKeypoints(img, keypoints, sift_feature_img);
    if (write_file) {
        std::string dir = "./ResultImgs/" + img_name;
        imwrite(dir + "_SIFT.jpg", sift_feature_img);
    }
    // cout << "SIFT size: " << sift_features.size << endl;
    sift_features.resize(2048);
    sift_features.convertTo(sift_features, CV_32F);
    sift_features = sift_features.reshape(1, 1);
    return sift_features;
}

Mat question3(const Mat& img, string img_name, bool write_file) {
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

    if (write_file) {
        string dir = "./ResultImgs/" + img_name;
        imwrite(dir + "_histogram.jpg", hist_img);
    }
    hist_img.convertTo(hist_img, CV_32F);
    hist_img = hist_img.reshape(1, 1);
    // cout << hist_img.size << std::endl;
    return hist_img;
}



void question4(Ptr<StatModel> model, Mat train_data, Mat train_label) {
    Ptr<TrainData> data = TrainData::create(train_data, ml::ROW_SAMPLE, train_label);

    data->shuffleTrainTest();
    model->train(data);
    
    
}

void question5(Ptr<StatModel> model, Ptr<TrainData> test_ptr, Mat test_data, Mat test_label) {
    Mat predicted = test_label.clone();
    // auto test_result = model->calcError(test_ptr, false, predicted);
    // cout << "Accuracy on test: " <<  test_result << "%\n";

    int false_positive = 0, false_negative = 0;
    model->predict(test_data, predicted);
    for (int i = 0; i < test_data.rows; ++i) {
        // cout << "Predicted = " << predicted.at<int>(i, 0) << "; \tactual = " << test_label.at<int>(i) << endl;
        if (predicted.at<int>(i, 0) != test_label.at<int>(i)) {
            false_positive++;
        }
    }
    cout << "Accuracy: " << 100 * (1.0 - ((double) false_positive / test_data.rows)) << "%" << std::endl;
    cout << "False positive: " << 100 * (((double) false_positive / test_data.rows)) << "%" << std::endl;
    cout << "False negative: " << 100 * (((double) false_positive / test_data.rows)) << "%" << std::endl;
}

void prepare_data(Mat& train_data, Mat& train_label, Mat& test_data, Mat& test_label, int question) {
    Mat img;
    std::string parent = "./ProjData/Train/bedroom";
    std::string img_name;
    bool write_file = true;
    for (const auto & entry : std::filesystem::directory_iterator{parent}) {
        // read as grayscale
        img = imread(entry.path(), IMREAD_GRAYSCALE); 
        img_name = entry.path().stem();
        // question1(img, img_name, write_file);
        // question2(img, img_name, write_file);
        if (question == 1) {
            train_data.push_back(question1(img, img_name, write_file));
        } 
        else if (question == 2) {
            train_data.push_back(question2(img, img_name, write_file));
        }
        else if (question == 3) {
            train_data.push_back(question3(img, img_name, write_file));
        }
        Mat label = Mat::ones(1, 1, CV_32F);
        label.at<int>(0) = 0;
        train_label.push_back(label); // bedroom
    }

    parent = "./ProjData/Train/Coast";
    for (const auto & entry : std::filesystem::directory_iterator{parent}) {
        // read as grayscale
        img = imread(entry.path(), IMREAD_GRAYSCALE); 
        img_name = entry.path().stem();
        // question1(img, img_name, write_file);
        // question2(img, img_name, write_file);
        if (question == 1) {
            train_data.push_back(question1(img, img_name, write_file));
        } 
        else if (question == 2) {
            train_data.push_back(question2(img, img_name, write_file));
        }
        else if (question == 3) {
            train_data.push_back(question3(img, img_name, write_file));
        }
        Mat label = Mat::ones(1, 1, CV_32F);
        label.at<int>(0) = 1;
        train_label.push_back(label); 
        // cout << "Train Label for coast: " << train_label.at<int>(train_label.rows -2, 0) << endl;
    }

    parent = "./ProjData/Train/Forest";
    for (const auto & entry : std::filesystem::directory_iterator{parent}) {
        // read as grayscale
        img = imread(entry.path(), IMREAD_GRAYSCALE); 
        img_name = entry.path().stem();
        // question1(img, img_name, write_file);
        // question2(img, img_name, write_file);
        if (question == 1) {
            train_data.push_back(question1(img, img_name, write_file));
        } 
        else if (question == 2) {
            train_data.push_back(question2(img, img_name, write_file));
        }
        else if (question == 3) {
            train_data.push_back(question3(img, img_name, write_file));
        }
        Mat label = Mat::ones(1, 1, CV_32F);
        label.at<int>(0) = 2;
        train_label.push_back(label);
    }


    // Get test data
    for (const auto & entry : std::filesystem::directory_iterator{ "./ProjData/Test/bedroom" }) {
        img = imread(entry.path(), IMREAD_GRAYSCALE);
        adjust_brightness(img);
        if (question == 1) {
            test_data.push_back(question1(img, "", false));
        } 
        else if (question == 2) {
            test_data.push_back(question2(img, "", false));
        }
        else if (question == 3) {
            test_data.push_back(question3(img, "", false));
        }
        Mat label = Mat::ones(1, 1, CV_32F);
        label.at<int>(0) = 0;
        test_label.push_back(label);
    }
    for (const auto & entry : std::filesystem::directory_iterator{ "./ProjData/Test/coast" }) {
        img = imread(entry.path(), IMREAD_GRAYSCALE);
        adjust_brightness(img);
        if (question == 1) {
            test_data.push_back(question1(img, "", false));
        } 
        else if (question == 2) {
            test_data.push_back(question2(img, "", false));
        }
        else if (question == 3) {
            test_data.push_back(question3(img, "", false));
        }
        Mat label = Mat::ones(1, 1, CV_32F);
        label.at<int>(0) = 1;
        test_label.push_back(label);
    }
    for (const auto & entry : std::filesystem::directory_iterator{ "./ProjData/Test/forest" }) {
        img = imread(entry.path(), IMREAD_GRAYSCALE);
        adjust_brightness(img);
        if (question == 1) {
            test_data.push_back(question1(img, "", false));
        } 
        else if (question == 2) {
            test_data.push_back(question2(img, "", false));
        }
        else if (question == 3) {
            test_data.push_back(question3(img, "", false));
        }
        Mat label = Mat::ones(1, 1, CV_32F);
        label.at<int>(0) = 2;
        test_label.push_back(label);
    }

}



int main() {
    // iterate over the files
    // std::string parent = "/Users/dk/Documents/School/CS4391/Assignments/TermProject/ProjData/Test/bedroom";
    Mat train_data;
    Mat train_label;
    Mat test_data;
    Mat test_label;

    Ptr<StatModel> model;
    Ptr<TrainData> test_ptr;
    cout << "\n\nQuestion 4A and 5: \n";
    // QUESTION 4A
    prepare_data(train_data, train_label, test_data, test_label, 1);
    model = KNearest::create();
    test_ptr = TrainData::create(test_data, ml::ROW_SAMPLE, test_label);
    question4(model, train_data, train_label);
    question5(model, test_ptr, test_data, test_label);

    // cout << "NEW MODEL" << endl;
    // model = KNearest::create();
    // question5(model, test_ptr, test_data, test_label);
    // cout << "DONE" << endl;

    train_data.release();
    train_label.release();
    test_data.release();
    test_label.release();
    train_data = Mat();
    train_label = Mat();
    test_data = Mat();
    test_label = Mat();

    cout << "\n\nQuestion 4B and 5: \n";
    // QUESTION 4B
    prepare_data(train_data, train_label, test_data, test_label, 2);
    cout << "done prep data \n";
    model = KNearest::create();
    test_ptr = TrainData::create(test_data, ml::ROW_SAMPLE, test_label);
    question4(model, train_data, train_label);
    question5(model, test_ptr, test_data, test_label);


    train_data.release();
    train_label.release();
    test_data.release();
    test_label.release();
    train_data = Mat();
    train_label = Mat();
    test_data = Mat();
    test_label = Mat();
    cout << "\n\nQuestion 4C and 5: \n";
    // QUESTION 4C
    prepare_data(train_data, train_label, test_data, test_label, 3);
    model = KNearest::create();
    test_ptr = TrainData::create(test_data, ml::ROW_SAMPLE, test_label);
    question4(model, train_data, train_label);
    question5(model, test_ptr, test_data, test_label);

    train_data.release();
    train_label.release();
    test_data.release();
    test_label.release();
    train_data = Mat();
    train_label = Mat();
    test_data = Mat();
    test_label = Mat();

    cout << "\n\nQuestion 4D and 5: \n";
    // QUESTION 4D
    Mat img;
    std::string parent = "./ProjData/Train/bedroom";
    std::string img_name;
    bool write_file = true;
    vector<int> train_label_vec;
    vector<int> test_label_vec;
    for (const auto & entry : std::filesystem::directory_iterator{parent}) {
        img = imread(entry.path(), IMREAD_GRAYSCALE); 
        img_name = entry.path().stem();
        train_data.push_back(question2(img, img_name, write_file));
        train_label_vec.push_back(0); // bedroom
    }

    parent = "./ProjData/Train/Coast";
    for (const auto & entry : std::filesystem::directory_iterator{parent}) {
        img = imread(entry.path(), IMREAD_GRAYSCALE); 
        img_name = entry.path().stem();
        train_data.push_back(question2(img, img_name, write_file));
        train_label_vec.push_back(1); 
    }

    parent = "./ProjData/Train/Forest";
    for (const auto & entry : std::filesystem::directory_iterator{parent}) {
        // read as grayscale
        img = imread(entry.path(), IMREAD_GRAYSCALE); 
        img_name = entry.path().stem();
        train_data.push_back(question2(img, img_name, write_file));
        train_label_vec.push_back(2);
    }


    // Get test data
    for (const auto & entry : std::filesystem::directory_iterator{ "./ProjData/Test/bedroom" }) {
        img = imread(entry.path(), IMREAD_GRAYSCALE);
        adjust_brightness(img);
        test_data.push_back(question2(img, "", false));
        test_label_vec.push_back(0);
    }
    for (const auto & entry : std::filesystem::directory_iterator{ "./ProjData/Test/coast" }) {
        img = imread(entry.path(), IMREAD_GRAYSCALE);
        adjust_brightness(img);
        test_data.push_back(question2(img, "", false));
        test_label_vec.push_back(1);
    }
    for (const auto & entry : std::filesystem::directory_iterator{ "./ProjData/Test/forest" }) {
        img = imread(entry.path(), IMREAD_GRAYSCALE);
        adjust_brightness(img);
        test_data.push_back(question2(img, "", false));
        test_label_vec.push_back(2);
    }
    train_label = Mat(train_label_vec, true);
    test_label = Mat(test_label_vec, true);
    Ptr<SVM> svm_model;
    svm_model = SVM::create();
    svm_model->setC(100);
    svm_model->setKernel(svm_model->LINEAR);
    svm_model->setType(svm_model->C_SVC);
    // svm_model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    test_ptr = TrainData::create(test_data, ml::ROW_SAMPLE, test_label);
    Ptr<TrainData> train_ptr = TrainData::create(train_data, ml::ROW_SAMPLE, train_label);
    svm_model->train(train_ptr);
    // question4(svm_model, train_data, train_label);
    Mat predicted = test_label.clone();
    int false_positive = 0, false_negative = 0;
    svm_model->predict(test_data, predicted);
    for (int i = 0; i < test_data.rows; ++i) {
        int predicted_val = predicted.at<int>(i, 0);
        if (predicted_val == 1065353216) { predicted_val = 1; }  // due to casting from float to int, causing undefined behaviors 
        if (predicted_val == 1073741824) { predicted_val = 2; }  // due to casting from float to int, causing undefined behaviors 
        
        if (predicted_val != test_label.at<int>(i)) {
            false_positive++;
        }
    }
    cout << "Accuracy: " << 100 * (1.0 - ((double) false_positive / test_data.rows)) << "%" << std::endl;
    cout << "False positive: " << 100 * (((double) false_positive / test_data.rows)) << "%" << std::endl;
    cout << "False negative: " << 100 * (((double) false_positive / test_data.rows)) << "%" << std::endl;
    return 0;
}
