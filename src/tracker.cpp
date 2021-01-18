#include <iostream>
#include "ros/ros.h"
#include "ros/package.h"
#include "tracking_object.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <queue>

#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Float32MultiArray.h"

#include <nav_msgs/Odometry.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/LinearMath/Transform.h>
#include <tf/LinearMath/Vector3.h>
#include "geometry_msgs/Point32.h"

#include "vehicle_detection/tracker_input.h"

using namespace std;

const int MAX_V = 100;
const int S = MAX_V - 2;
const int T = MAX_V - 1;
const int WORK = 40;
const int INF = 987654321;
const double DIST_THRESHOLD = 6.0;


struct Pose{
    double x, y, z;
};

struct Orientation{
    double x, y, z, w;
};

struct EulerAngles {
    double roll, pitch, yaw;
};

struct CarState{
    Pose pose;
    Orientation orientation;
};

EulerAngles ToEulerAngles(Orientation q) {
    EulerAngles angles;

    double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    angles.roll = std::atan2(sinr_cosp, cosr_cosp);

    double sinp = 2 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
        angles.pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        angles.pitch = std::asin(sinp);

    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    angles.yaw = std::atan2(siny_cosp, cosy_cosp);

    return angles;
}

Orientation ToQuaternion(double yaw, double pitch, double roll)
{
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    Orientation q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q;
}

class Tracker{
    private:
        vector<TrackingObject> tracking_objects;
        ros::NodeHandle nh;
        ros::Subscriber tracker_input_sub;
        ros::Subscriber Odometry_sub;
        CarState car_state;
        double prev_time;
        bool is_first, car_state_init;

    public:
        Tracker();
        void tracker_input_callback(const vehicle_detection::tracker_input::ConstPtr& msg);
        void Odometry_callback(const nav_msgs::Odometry::ConstPtr& msg);
};

Tracker::Tracker(){
    tracker_input_sub = nh.subscribe("/DL_result", 1, &Tracker::tracker_input_callback, this);
    Odometry_sub = nh.subscribe("/Odometry", 1, &Tracker::Odometry_callback, this);
    
    car_state.pose.x = 0;
    car_state.pose.y = 0;
    car_state.pose.z = 0;
    car_state.orientation.x = 0;
    car_state.orientation.y = 0;
    car_state.orientation.z = 0;
    car_state.orientation.w = 0;

    prev_time = 0;
    is_first = true;
    car_state_init = false;
}

void Tracker::Odometry_callback(const nav_msgs::Odometry::ConstPtr& msg){
    car_state.pose.x = msg->pose.pose.position.x;
    car_state.pose.y = msg->pose.pose.position.y;
    car_state.pose.z = msg->pose.pose.position.z;
    car_state.orientation.x = msg->pose.pose.orientation.x;
    car_state.orientation.y = msg->pose.pose.orientation.y;
    car_state.orientation.z = msg->pose.pose.orientation.z;
    car_state.orientation.w = msg->pose.pose.orientation.w;

    car_state_init = true;
}

void Tracker::tracker_input_callback(const vehicle_detection::tracker_input::ConstPtr& msg)
{
    int input_num = static_cast<int>((msg->size)/6);
    int tracking_objs_num = tracking_objects.size();
    double delta_t = msg->header.stamp.sec + (msg->header.stamp.nsec/1.0e9) - prev_time;
    prev_time = msg->header.stamp.sec + msg->header.stamp.nsec/1.0e9;

    if(car_state_init){
        tf::Transform car_tf;
        car_tf.setOrigin(tf::Vector3(car_state.pose.x, car_state.pose.y, car_state.pose.z));
        car_tf.setRotation(tf::Quaternion(car_state.orientation.x, car_state.orientation.y, car_state.orientation.z, car_state.orientation.w));

        vector<vector<double>> detected_objs;
        vector<double> temp;
        detected_objs.clear();

        tf::Transform obj_tf, tf;
        tf::Vector3 obj_origin;
        tf::Quaternion obj_rotation;
        double roll, pitch, yaw;
        for(int i=0; i<input_num; i++){
            obj_tf.setOrigin(tf::Vector3(msg->data[6*i + 2], msg->data[6*i + 1], 0));
            obj_tf.setRotation(tf::Quaternion(msg->data[i + 5], 0, 0));
            tf = car_tf * obj_tf;
            obj_origin = tf.getOrigin();
            obj_rotation = tf.getRotation();
            tf::Matrix3x3 m(obj_rotation);
            m.getRPY(roll, pitch, yaw);
            
            temp.clear();
            temp.push_back(obj_origin.x());
            temp.push_back(obj_origin.y());
            temp.push_back(yaw);
            temp.push_back(msg->data[6*i + 0]);
            temp.push_back(msg->data[6*i + 4]);
            temp.push_back(msg->data[6*i + 3]);

            detected_objs.push_back(temp);
        }

        if(is_first && (input_num != 0)){
            for(int i=0; i<input_num; i++){
                TrackingObject tmp(detected_objs[i][0], detected_objs[i][1], detected_objs[i][2], static_cast<int>(detected_objs[i][3]), detected_objs[i][4], detected_objs[i][5]);
                tracking_objects.push_back(tmp);
            }
            is_first = false;
        }
        else{
            vector<vector<double>> tracking_objs_st;
            tracking_objs_st.clear();

            for(int i=0; i<tracking_objs_num; i++){
                tracking_objects[i].calc_st_pred(delta_t);
                tracking_objs_st.push_back(tracking_objects[i].return_st_pred());
            }

            // MCMF start
            vector<int> adj[MAX_V];
            double c[MAX_V][MAX_V] = {0.,};
            double f[MAX_V][MAX_V] = {0.,};
            double d[MAX_V][MAX_V] = {0.,};

            for(int i=0; i<input_num; i++){
                c[S][i] = 1;

                adj[S].push_back(i);
                adj[i].push_back(S);
            }

            for(int i=0; i<tracking_objs_num; i++){
                c[i + WORK][T] = 1;

                adj[i + WORK].push_back(T);
                adj[T].push_back(i + WORK);
            }

            for(int i=0; i<input_num; i++){
                for(int j=0; j<tracking_objs_num; j++){
                    double cost;
                    cost = static_cast<double>(sqrt(pow(tracking_objs_st[j][0]-detected_objs[i][0],2) + pow(tracking_objs_st[j][1]-detected_objs[i][1],2)));

                    if(cost < DIST_THRESHOLD){
                        adj[i].push_back(j + WORK);
                        adj[j + WORK].push_back(i);

                        d[i][j + WORK] = cost;
                        d[j + WORK][i] = -cost;

                        c[i][j + WORK] = 1;
                    }
                }
            }

            int *edge = new int[tracking_objs_num];
            int *input_check = new int[input_num]();

            for(int i=0; i<tracking_objs_num; i++){
                edge[i] = -1;
            }

            while(1)
            {
                int prev[MAX_V], dist[MAX_V];
                bool inQ[MAX_V] = {0};

                queue<int> q;
                fill(prev, prev + MAX_V, -1);
                fill(dist, dist + MAX_V, INF);

                dist[S] = 0;
                inQ[S] = true;

                q.push(S);

                while(!q.empty())
                {
                    int here = q.front();
                    q.pop();

                    inQ[here] = false;

                    for(int i=0; i < adj[here].size(); i++)
                    {
                        int next = adj[here][i];
                        if (c[here][next] - f[here][next] > 0 && dist[next] > dist[here] + d[here][next])
                        {
                            dist[next] = dist[here] + d[here][next];
                            prev[next] = here;
                            if (!inQ[next])
                            {
                                q.push(next);
                                inQ[next] = true;
                            }
                        }
                    }
                }

                if (prev[T] == -1){
                    break;
                }

                double flow = INF;
                for (int i = T; i != S; i = prev[i]){
                    flow = min(flow, c[prev[i]][i] - f[prev[i]][i]);
                }
        
                for (int i = T; i != S; i = prev[i])
                {
                    f[prev[i]][i] += flow;
                    f[i][prev[i]] -= flow;
                }
                
                edge[prev[T] - WORK] = prev[prev[T]];
                input_check[prev[prev[T]]] = 1;
            }
            //MCMF end
            
            for(int i=0; i<tracking_objs_num; i++){
                if(edge[i] != -1){
                    int idx = edge[i];
                    tracking_objects[i].calc_P_k1_k(delta_t);
                    tracking_objects[i].calc_kalman_gain();
                    tracking_objects[i].calc_st(detected_objs[idx][0], detected_objs[idx][1], detected_objs[idx][2]);
                    tracking_objects[i].calc_P();
                    tracking_objects[i].count = 0;
                }
            }

            for(int i=0; i<tracking_objs_num; i++){
                if(edge[i] == -1){
                    if(tracking_objects[i].count >= 10){
                        tracking_objects.erase(tracking_objects.begin() + i);
                    }
                    else{
                        tracking_objects[i].count++;
                        tracking_objects[i].sub_st_pred_to_st();
                    }
                }
            }

            for(int i=0; i<input_num; i++){
                if(input_check[i] == 0){
                    TrackingObject tmp(detected_objs[i][0], detected_objs[i][1], detected_objs[i][2], static_cast<int>(detected_objs[i][3]), detected_objs[i][4], detected_objs[i][5]);
                    tracking_objects.push_back(tmp);
                }
            }

            delete[] edge;
            delete[] input_check;
        }

        std::cout <<"Number of tracking objects : " << tracking_objects.size() <<std::endl;
        for(int i=0; i<tracking_objects.size(); i++){
            vector<double> vis = tracking_objects[i].return_st_pred();
            std::cout << vis[0] << " " << vis[1] << std::endl;
        }
    }
}

int main(int argc, char ** argv)
{
    ros::init(argc, argv, "vehicle_tracker");

    Tracker tracker;
    
    ros::spin();

    return 0;
}