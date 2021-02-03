#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <time.h>

#include "Eigen/Core"
#include "Eigen/Geometry"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include "ros/ros.h"
#include "ros/package.h"

#include "Darknet.h"
#include "Visualize.h"
#include "NMS.h"

#include <sensor_msgs/PointCloud.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/LinearMath/Transform.h>
#include <tf/LinearMath/Vector3.h>
#include "geometry_msgs/Point32.h"
#include "tf_conversions/tf_eigen.h"

#include "vehicle_detection/tracker_input.h"

#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

using namespace message_filters;
using namespace sensor_msgs;
using namespace std; 
using namespace std::chrono; 
using namespace cv;
using namespace Eigen;

const double cm_per_pixel = 5.0;
const double map_scale = 100.0/cm_per_pixel;
const int map_height = 800;
const int map_width = 800;
const int img_size = 800;
const double pi = M_PI;
const int img_cor_x = map_width/2;
const int img_cor_y = map_scale * 10;

string net_path = "/home/sunho/catkin_ws/src/vehicle_detection/custom.cfg";
torch::DeviceType device_type = torch::kCUDA;
torch::Device device(device_type);
Darknet net(net_path.c_str(), &device);

cv::Mat calc_homography(double x, double y, double z, double R, double P, double Y)
{
    cv::Mat homography;

    tf::Quaternion cam_q, VToCam;
    tf::Quaternion car_to_cam_q;
    tf::Transform car_to_cam_tf, final_tf, cam_tf;
    
    Eigen::Affine3d cam_affine;

    Matrix<double, 3, 4> mat_3x4;
    Matrix<double, 3, 3> vehicle_to_cam;
    Matrix<double, 3, 3> homo;
    Matrix<double, 3, 4> P_matrix;

    Matrix<double, 3, 3> dist_to_img;

    tf::Vector3 cam_v = tf::Vector3(0, 0, 0);
    cam_q.setRPY(-90.0*M_PI/180.0, 0, -90.0*M_PI/180.0);
    cam_tf = tf::Transform(cam_q, cam_v);
    
    P_matrix(0,0) = 346.43486958; P_matrix(0,1) = 0.0; P_matrix(0,2) = 600.0; P_matrix(0,3) = 0.0;
    P_matrix(1,0) = 0.0; P_matrix(1,1) = 346.43486958; P_matrix(1,2) = 400.0; P_matrix(1,3) = 0.0;
    P_matrix(2,0) = 0.0; P_matrix(2,1) = 0.0; P_matrix(2,2) = 1.0; P_matrix(2,3) = 0.0;
    
    dist_to_img.setZero();
    dist_to_img(0,1) = -map_scale; dist_to_img(0,2) = 20 * map_scale;
    dist_to_img(1,0) = -map_scale; dist_to_img(1,2) = 30 * map_scale;
    dist_to_img(2,2) = 1;
    
    tf::Vector3 car_to_cam_v = tf::Vector3(x, y, z);
    car_to_cam_q.setRPY(R*M_PI/180.0, P*M_PI/180.0, Y*M_PI/180.0);
    car_to_cam_tf = tf::Transform(car_to_cam_q, car_to_cam_v);
    final_tf = car_to_cam_tf * cam_tf;
    tf::transformTFToEigen(final_tf.inverse(), cam_affine);
    mat_3x4 = P_matrix * cam_affine.matrix();

    vehicle_to_cam(0,0) = mat_3x4(0,0); vehicle_to_cam(0,1) = mat_3x4(0,1); vehicle_to_cam(0,2) = mat_3x4(0,3);
    vehicle_to_cam(1,0) = mat_3x4(1,0); vehicle_to_cam(1,1) = mat_3x4(1,1); vehicle_to_cam(1,2) = mat_3x4(1,3);
    vehicle_to_cam(2,0) = mat_3x4(2,0); vehicle_to_cam(2,1) = mat_3x4(2,1); vehicle_to_cam(2,2) = mat_3x4(2,3);
    
    homo = dist_to_img * vehicle_to_cam.inverse();
    cv::eigen2cv(homo,homography);
            
    return homography;
}
class DataSubscriber{
    private:
        ros::NodeHandle nh;
        message_filters::Subscriber<sensor_msgs::PointCloud> pc_sub1;
        message_filters::Subscriber<sensor_msgs::PointCloud> pc_sub2;
        message_filters::Subscriber<sensor_msgs::Image> img_1;
        message_filters::Subscriber<sensor_msgs::Image> img_2;
        message_filters::Subscriber<sensor_msgs::Image> img_3;
        typedef sync_policies::ApproximateTime<PointCloud, PointCloud, Image, Image, Image> SyncPolicy;
        typedef Synchronizer<SyncPolicy> Sync;
        boost::shared_ptr<Sync> sync;
        tf::Transform lidar_tf_1;
        tf::Transform lidar_tf_2;      
        tf::Quaternion id;
        int input_image_size;  
        float iou_threshold, conf_threshold, class_threshold;
        ros::Publisher pub;
        cv::Mat homography_front, homography_right, homography_left;
        cv::Mat front_mask, right_mask, left_mask;

    public:
        DataSubscriber(){
            pc_sub1.subscribe(nh, "/pointcloud/vlp", 3);
            pc_sub2.subscribe(nh, "/pointcloud/os1", 3);
            img_1.subscribe(nh, "/color1/image_raw", 3);
            img_2.subscribe(nh, "/color2/image_raw", 3);
            img_3.subscribe(nh, "/color3/image_raw", 3);
            sync.reset(new Sync(SyncPolicy(3), pc_sub1, pc_sub2, img_1, img_2, img_3));
            sync->registerCallback(boost::bind(&DataSubscriber::callback, this, _1, _2, _3, _4, _5));

            tf::Quaternion q_1;
            tf::Vector3 v_1 = tf::Vector3(2.0, 0.0, 2.0);
            q_1.setRPY(0,0,90.0*M_PI/180.0);
            lidar_tf_1 = tf::Transform(q_1,v_1);

            tf::Quaternion q_2;
            tf::Vector3 v_2 = tf::Vector3(2.5, 0.0, 2.0);
            q_2.setRPY(0,0,0);
            lidar_tf_2 = tf::Transform(q_2,v_2);

            id.setRPY(0,0,0);

            input_image_size = 416;

            ros::param::get("/iou_threshold", iou_threshold);
            ros::param::get("/conf_threshold", conf_threshold);
            ros::param::get("/class_threshold", class_threshold);

            pub = nh.advertise<vehicle_detection::tracker_input>("/DL_result", 1);

            homography_right = calc_homography(2.1,-0.7, 2.6, 0, 25.0, -100.0);
            homography_front = calc_homography(2.3, 0.0, 2.6, 0, -15.0, 0);
            homography_left = calc_homography(2.1, 0.7, 2.6, 0, 25.0, 100.0);

            stringstream front_mask_path, right_mask_path, left_mask_path;
            front_mask_path << ros::package::getPath("vehicle_detection") << "/mask_img/front_mask.png";
            right_mask_path << ros::package::getPath("vehicle_detection") << "/mask_img/right_mask.png";
            left_mask_path << ros::package::getPath("vehicle_detection") << "/mask_img/left_mask.png";

            front_mask = cv::imread(front_mask_path.str(), 0);
            right_mask = cv::imread(right_mask_path.str(), 0);
            left_mask = cv::imread(left_mask_path.str(), 0);
        }

        void callback(const PointCloud::ConstPtr& pc_msg1, const PointCloud::ConstPtr& pc_msg2, const Image::ConstPtr& img_msg1, const Image::ConstPtr& img_msg2, const Image::ConstPtr& img_msg3);
        geometry_msgs::Point32 vector_to_point(tf::Vector3 v);
};

void DataSubscriber::callback(const PointCloud::ConstPtr& pc_msg1, const PointCloud::ConstPtr& pc_msg2, const Image::ConstPtr& img_msg1, const Image::ConstPtr& img_msg2, const Image::ConstPtr& img_msg3){
    auto start = std::chrono::high_resolution_clock::now();

    // Point Cloud Data
    Mat BEV_map(map_height, map_width, CV_32FC3, Scalar(0.0, 0.0, 0.0));
    double z_min = 98765432.0;

    for(geometry_msgs::Point32 point : pc_msg1->points){
        geometry_msgs::Point32 new_point;
        tf::Vector3 p = tf::Vector3(point.x, point.y, point.z);
        tf::Transform m = lidar_tf_1 * tf::Transform(id, p);
        tf::Vector3 v = m.getOrigin();
        new_point = vector_to_point(v);
        if (!((new_point.x >= 0) && (new_point.x <= 4.5) && (new_point.y >= -1) && (new_point.y <= 1)) && (new_point.z <= 3))
        {
            if ((new_point.x * map_scale + img_cor_y >= 0) && (new_point.x * map_scale + img_cor_y < map_height)){
                if((new_point.y * map_scale + img_cor_x >= 0) && (new_point.y * map_scale + img_cor_x < map_width)){
                    int y = (map_height - 1) - static_cast<int>(new_point.x * map_scale + img_cor_y);
                    int x = (map_width - 1) - static_cast<int>(new_point.y * map_scale + img_cor_x);
                    if (BEV_map.at<cv::Vec3f>(y,x)[2] == 0.0){
                        BEV_map.at<cv::Vec3f>(y,x)[2] = 1.0;
                    }
                    if (BEV_map.at<cv::Vec3f>(y,x)[0] < new_point.z){
                        BEV_map.at<cv::Vec3f>(y,x)[0] = new_point.z;
                    }
                    if (new_point.z < z_min){
                        z_min = new_point.z;
                    }
                }
            }
        }
    }

    for(geometry_msgs::Point32 point : pc_msg2->points){
        geometry_msgs::Point32 new_point;
        tf::Vector3 p = tf::Vector3(point.x, point.y, point.z);
        tf::Transform m = lidar_tf_2 * tf::Transform(id, p);
        tf::Vector3 v = m.getOrigin();
        new_point = vector_to_point(v);

        if (!((new_point.x >= 0) && (new_point.x <= 4.5) && (new_point.y >= -1) && (new_point.y <= 1)) && (new_point.z <= 3))
        {
            if ((new_point.x * map_scale + img_cor_y >= 0) && (new_point.x * map_scale + img_cor_y < map_height)){
                if((new_point.y * map_scale + img_cor_x >= 0) && (new_point.y * map_scale + img_cor_x < map_width)){
                    int y = (map_height - 1) - static_cast<int>(new_point.x * map_scale + img_cor_y);
                    int x = (map_width - 1) - static_cast<int>(new_point.y * map_scale + img_cor_x);
                    if (BEV_map.at<cv::Vec3f>(y,x)[2] == 0.0){
                        BEV_map.at<cv::Vec3f>(y,x)[2] = 1.0;
                    }
                    if (BEV_map.at<cv::Vec3f>(y,x)[0] < new_point.z){
                        BEV_map.at<cv::Vec3f>(y,x)[0] = new_point.z;
                    }
                    if (new_point.z < z_min){
                        z_min = new_point.z;
                    }
                }
            }
        }
    }

    double z_max = z_min + 3.5;
    
    for(int i = 0; i < map_width; i++){
        for(int j = 0; j < map_height; j++){
            if(BEV_map.at<cv::Vec3f>(j,i)[2] == 1.0){
                BEV_map.at<cv::Vec3f>(j,i)[0] = (BEV_map.at<cv::Vec3f>(j,i)[0] - z_min) / (z_max - z_min);
                if(BEV_map.at<cv::Vec3f>(j,i)[0] > 1.0){
                    // BEV_map.at<cv::Vec3f>(j,i)[0] = 0.0;
                    // BEV_map.at<cv::Vec3f>(j,i)[2] = 0.0;
                    BEV_map.at<cv::Vec3f>(j,i)[0] = 1.0;
                }
            }
        }
    }

    //cv::imshow("BEV_map", BEV_map);

    // Camera Data
    cv::Mat Image_BEV_map(map_height, map_width, CV_32FC3, Scalar(0.0, 0.0, 0.0));
    cv::Mat Camera_BEV_left(map_height, map_width, CV_32FC3, Scalar(0.0, 0.0, 0.0));
    cv::Mat Camera_BEV_front(map_height, map_width, CV_32FC3, Scalar(0.0, 0.0, 0.0));
    cv::Mat Camera_BEV_right(map_height, map_width, CV_32FC3, Scalar(0.0, 0.0, 0.0));

    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(img_msg1, sensor_msgs::image_encodings::BGR8);
    cv::warpPerspective(cv_ptr->image,Camera_BEV_front,homography_front, cv::Size(map_height, map_width));
    
    cv_ptr = cv_bridge::toCvCopy(img_msg2, sensor_msgs::image_encodings::BGR8);
    cv::warpPerspective(cv_ptr->image,Camera_BEV_right,homography_right, cv::Size(map_height, map_width));
    
    cv_ptr = cv_bridge::toCvCopy(img_msg3, sensor_msgs::image_encodings::BGR8);
    cv::warpPerspective(cv_ptr->image,Camera_BEV_left,homography_left, cv::Size(map_height, map_width));

    Camera_BEV_right.copyTo(Image_BEV_map, right_mask);
    Camera_BEV_left.copyTo(Image_BEV_map, left_mask);
    Camera_BEV_front.copyTo(Image_BEV_map, front_mask);

    //imshow("Image_BEV_map", Image_BEV_map);
    
    // Architecture Start
    cv::Mat resized_image, Camera_resized_image;

    cv::cvtColor(BEV_map, resized_image, cv::COLOR_BGR2RGB);
    cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

    cv::cvtColor(Image_BEV_map, Camera_resized_image, cv::COLOR_BGR2RGB);
    cv::resize(Camera_resized_image, Camera_resized_image, cv::Size(input_image_size, input_image_size));
    
    cv::Mat camera_float;
    Camera_resized_image.convertTo(camera_float, CV_32FC3, 1.0/255.0);

    auto img_tensor = torch::from_blob(resized_image.data, {1, input_image_size, input_image_size, 3}).to(device);
    auto camera_tensor = torch::from_blob(camera_float.data, {1, input_image_size, input_image_size, 3}).to(device);

    auto input_tensor = torch::cat({img_tensor, camera_tensor}, 3);
    input_tensor = input_tensor.permute({0,3,1,2});

    auto output = net.forward(input_tensor).to(torch::kCPU);
    
    // filter result by NMS

    int arr_size = output.sizes()[1];
    float* output_arr = output.data_ptr<float>();

    vector<vector<float>> result;
    result = non_maximum_suppresion(output_arr, arr_size, conf_threshold, iou_threshold, class_threshold);

    int result_size = result.size();
    for(int i=0; i<result_size; i++)
    {
        DrawRotatedRectangle(resized_image, Point2f(result[i][1], result[i][2]), Size2f(result[i][4], result[i][3]), -result[i][5]*360/pi);
    }

    cv::imshow("img", resized_image);
    cv::waitKey(1);

    vehicle_detection::tracker_input msg;

    // class, y, x, h, w, rad
    msg.size = result.size() * 6;
    msg.header = pc_msg1->header;
   
    std::vector<float> vec(result.size() * 6, 0);
    
    float resize_factor = static_cast<float>(img_size) / static_cast<float>(input_image_size);

    for (int i=0; i<result.size(); i++){
        for (int j=0; j<6; j++){
            if (j == 1){
                vec[i*6 + j] = cm_per_pixel * resize_factor *  (static_cast<float>(img_cor_x / resize_factor) - result[i][j]) / 100.0;
            }
            else if (j == 2){
                vec[i*6 + j] = cm_per_pixel * resize_factor *  (input_image_size - img_cor_y / resize_factor - result[i][j]) / 100.0;
            }
            else if (j == 3 || j == 4){
                vec[i*6 + j] = cm_per_pixel * resize_factor * result[i][j] / 100.0;
            }
            else if (j == 5){
                float temp_rad = result[i][j] - M_PI_2;
                if(temp_rad < -M_PI){
                    temp_rad += 2 * M_PI;
                }
                vec[i*6 + j] = temp_rad;
            }
            else{
                vec[i*6 + j] = result[i][j];
            }
        }
    }       
    
    msg.data = vec;

    pub.publish(msg);

    result.clear();
    vector<vector<float>>(result).swap(result);
    
    BEV_map.release();
    //resized_image.release();

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start); 

    std::cout << "inference taken : " << duration.count() << " ms" << endl; 

}

geometry_msgs::Point32 DataSubscriber::vector_to_point(tf::Vector3 v){
    geometry_msgs::Point32 p;
    p.x = v.x();
    p.y = v.y();
    p.z = v.z();
    return p;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "vehicle_detection_pc");

    stringstream weight_path;
    weight_path << ros::package::getPath("vehicle_detection") << "/newYolov3.weights";
    string weight_path_str = weight_path.str();

    int input_image_size = 416;

    map<string, string> *info = net.get_net_info();

    info->operator[]("height") = std::to_string(input_image_size);

    std::cout << "loading weight ..." << endl;
    net.load_weights(weight_path_str.c_str());
    std::cout << "weight loaded ..." << endl;
    
    net.to(device);

    torch::NoGradGuard no_grad;
    net.eval();

    std::cout << "start to inference ..." << endl;

    DataSubscriber data_subscriber;

    ros::spin();

    return 0;
}
