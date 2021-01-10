#include <NMS.h>

const double pi = 3.14159265358979f;

vector<vector<float>> non_maximum_suppresion(float* output_arr, int size, float conf_threshold, float iou_threshold, float class_threshold)
{
    int box_num = 0;
    vector<vector<float>> result;
    result.clear();

    for(int i=0; i<size; i++)
    {
        if(output_arr[i*10 + 4] > conf_threshold)
        {
            vector<float> temp;
            int max_idx = max_class(output_arr[i*10 + 6], output_arr[i*10 + 7], output_arr[i*10 + 8], output_arr[i*10 + 9], class_threshold);
            if(max_idx != -1)
            {
                temp.push_back(static_cast<float>(max_idx));
                temp.push_back(output_arr[i*10 + 0]);
                temp.push_back(output_arr[i*10 + 1]);
                temp.push_back(output_arr[i*10 + 2]);
                temp.push_back(output_arr[i*10 + 3]);
                temp.push_back(output_arr[i*10 + 5]);
                temp.push_back(output_arr[i*10 + 4]);

                if(box_num == 0)
                {
                    result.push_back(temp);
                    box_num++;
                }
                else
                {
                    int num = box_num;
                    bool check = false;

                    for(int j=0; j<num; j++){
                        if(calc_iou(result[j],temp) > iou_threshold && result[j][0] == temp[0]){
                            if(result[j][6] < temp[6])
                            {
                                result.erase(result.begin() + j);
                                result.push_back(temp);
                            }
                            check = true;
                            break;
                        }
                    }
                    if(check == false){
                        result.push_back(temp);
                        box_num++;
                    }
                }
            }
            temp.clear();
            vector<float>(temp).swap(temp);
        }
    }
    return result;
}

int max_class(float class_1, float class_2, float class_3, float class_4, float class_threshold)
{
    if((class_1 > class_2) && (class_1 > class_3) && (class_1 > class_4) && (class_1 > class_threshold)){
        return 0;
    }
    else if((class_2 > class_1) && (class_2 > class_3) && (class_2 > class_4) && (class_2 > class_threshold)){
        return 1;
    }
    else if((class_3 > class_2) && (class_3 > class_1) && (class_3 > class_4) && (class_3 > class_threshold)){
        return 2;
    }
    else if((class_4 > class_2) && (class_4 > class_3) && (class_4 > class_1) && (class_4 > class_threshold)){
        return 3;
    }
    else{
        return -1;
    }
}

double calc_iou(vector<float> box_1, vector<float> box_2)
{   
    cv::Point2f center_1(box_1[1], box_1[2]);
    cv::Point2f center_2(box_2[1], box_2[2]);
    cv::Size2f size_1(box_1[4], box_1[3]);
    cv::Size2f size_2(box_2[4], box_2[3]);

    cv::RotatedRect rotatedRec_1(center_1, size_1, -static_cast<double>(box_1[5]*360/pi));
    cv::RotatedRect rotatedRec_2(center_2, size_2, -static_cast<double>(box_2[5]*360/pi));

    vector<cv::Point2f> vertices, vertices_sorted;
    int type = cv::rotatedRectangleIntersection(rotatedRec_1, rotatedRec_2, vertices);

    double inter_area;
    if(type == 0){
        inter_area = 0;
    }
    else{
        cv::convexHull(vertices, vertices_sorted);
        inter_area = cv::contourArea(vertices_sorted);
    }
    
    double total_area = static_cast<double>(box_1[4]*box_1[3]) + static_cast<double>(box_2[4]*box_2[3]) - inter_area;

    vertices.clear();
    vector<cv::Point2f>(vertices).swap(vertices);
    vertices_sorted.clear();
    vector<cv::Point2f>(vertices_sorted).swap(vertices_sorted);

    return inter_area/total_area;
}