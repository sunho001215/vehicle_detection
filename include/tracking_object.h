#include <iostream>
#include <Eigen/Eigen>
#include <cmath>
#include "ros/ros.h"
#include "ros/package.h"

using namespace Eigen;

#define _st_DIM 8
#define _Z_DIM 3

class TrackingObject{
    public:
        Matrix<double, _st_DIM, 1> st_k;
        Matrix<double, _st_DIM, 1> st_k1_k;
        Matrix<double, _st_DIM, _st_DIM> F;
        Matrix<double, _st_DIM, _st_DIM> Q;
        Matrix<double, _st_DIM, _st_DIM> P_k;
        Matrix<double, _st_DIM, _st_DIM> P_k1_k;
        Matrix<double, _Z_DIM, _st_DIM> H;
        Matrix<double, _Z_DIM, _Z_DIM> S;
        Matrix<double, _Z_DIM, _Z_DIM> R;
        Matrix<double, _st_DIM, _Z_DIM> K;

    private:
        TrackingObject(double x, double y, double yaw);
        void calc_st_pred(double delta_t);
        void calc_P_k1_k(double delta_t);
        void calc_kalman_gain();
        void calc_st(double x, double y, double yaw);
        void calc_P();
};