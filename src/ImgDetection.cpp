#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ros/ros.h"
#include "ros/package.h"

#include "Darknet.h"
#include "Visualize.h"
#include "NMS.h"

using namespace std; 
using namespace std::chrono; 
using namespace cv;

const double pi = 3.14159265358979f;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "vehicle_detection_img");

    string img_path;
    ros::param::get("/img_path", img_path);
    std::cout << img_path << std::endl;

    stringstream net_path;
    net_path << ros::package::getPath("vehicle_detection") <<"/custom.cfg";
    string net_path_str = net_path.str();

    stringstream weight_path;
    weight_path << ros::package::getPath("vehicle_detection") << "/newYolov3.weights";
    string weight_path_str = weight_path.str();

    float iou_threshold, conf_threshold, class_threshold;
    ros::param::get("/iou_threshold", iou_threshold);
    ros::param::get("/conf_threshold", conf_threshold);
    ros::param::get("/class_threshold", class_threshold);

    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {        
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    // input image size for YOLO v3
    int input_image_size = 416;

    
    Darknet net(net_path_str.c_str(), &device);

    map<string, string> *info = net.get_net_info();

    info->operator[]("height") = std::to_string(input_image_size);

    std::cout << "loading weight ..." << endl;
    net.load_weights(weight_path_str.c_str());
    std::cout << "weight loaded ..." << endl;
    
    net.to(device);

    torch::NoGradGuard no_grad;
    net.eval();

    std::cout << "start to inference ..." << endl;
    
    cv::Mat origin_image, resized_image;

    origin_image = cv::imread(img_path.c_str());
    
    cv::cvtColor(origin_image, resized_image,  cv::COLOR_BGR2RGB);
    cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0/255.0);

    auto img_tensor = torch::from_blob(img_float.data, {1, input_image_size, input_image_size, 3}).to(device);
    img_tensor = img_tensor.permute({0,3,1,2});

    auto start = std::chrono::high_resolution_clock::now();
       
    auto output = net.forward(img_tensor).to(torch::kCPU);
    
    // filter result by NMS 

    int arr_size = output.sizes()[1];
    //std::cout << arr_size << std::endl;
    float* output_arr = output.data_ptr<float>();
    
    vector<vector<float>> result;
    result = non_maximum_suppresion(output_arr, arr_size, conf_threshold, iou_threshold, class_threshold);

    int result_size = result.size();
    std::cout << "Number of Object : " << result_size << std::endl;
    for(int i=0; i<result_size; i++)
    {
            DrawRotatedRectangle(resized_image, Point2f(result[i][1], result[i][2]), Size2f(result[i][4], result[i][3]), -result[i][5]*360/pi);
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start); 

    // It should be known that it takes longer time at first time
    std::cout << "inference taken : " << duration.count() << " ms" << endl; 

    cv::imshow("img", resized_image);
    cv::waitKey(0);
    
    return 0;
}
