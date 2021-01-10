#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ros/ros.h"
#include "ros/package.h"

using namespace std;
using namespace cv;

vector<vector<float>> non_maximum_suppresion(float* output_arr, int size, float conf_threshold, float iou_threshold, float class_threshold);
int max_class(float class_1, float class_2, float class_3, float class_4, float class_threshold);
double calc_iou(vector<float> box_1, vector<float> box_2);