#include "tracking_object.h"

TrackingObject::TrackingObject(double x, double y, double yaw)
{
    double initial_state_x_d, initial_state_y_d, initial_state_x_dd, initial_state_y_dd;
    double initial_state_yaw_rate;
    
    ros::param::get("/initial_state_x_d", initial_state_x_d);
    ros::param::get("/initial_state_y_d", initial_state_y_d);
    ros::param::get("/initial_state_x_dd", initial_state_x_dd);
    ros::param::get("/initial_state_y_dd", initial_state_y_dd);
    ros::param::get("/initial_state_yaw_rate", initial_state_yaw_rate);

    st_k(0,0) = x;
    st_k(1,0) = y;
    st_k(2,0) = initial_state_x_d;
    st_k(3,0) = initial_state_y_d;
    st_k(4,0) = initial_state_x_dd;
    st_k(5,0) = initial_state_y_dd;
    st_k(6,0) = yaw;
    st_k(7,0) = initial_state_yaw_rate;

    double initial_pose_error_std, initial_state_error_std;
    double pose_error_var, state_error_var;

    ros::param::get("/initial_pose_error_std", initial_pose_error_std);
    ros::param::get("/initial_state_error_std", initial_state_error_std);

    pose_error_var = pow(initial_pose_error_std, 2);
    state_error_var = pow(initial_state_error_std, 2);

    P_k.setZero();
    P_k(0,0) = pose_error_var;
    P_k(1,1) = pose_error_var;
    P_k(2,2) = state_error_var;
    P_k(3,3) = state_error_var;
    P_k(4,4) = state_error_var;
    P_k(5,5) = state_error_var;
    P_k(6,6) = pose_error_var;
    P_k(7,7) = state_error_var;

    H.setZero();
    H(0,0) = 1;
    H(1,1) = 1;
    H(2,6) = 1;
}

void TrackingObject::calc_st_pred(double delta_t)
{
    F.setIdentity();
    F(0,2) = delta_t;
    F(0,4) = pow(delta_t, 2) / 2.0;
    F(1,3) = delta_t;
    F(1,5) = pow(delta_t, 2) / 2.0;
    F(2,4) = delta_t;
    F(3,5) = delta_t;
    F(6,7) = delta_t;

    st_k1_k = F*st_k;
}

void TrackingObject::calc_P_k1_k(double delta_t)
{
    Q.setZero();

    double system_noise_a_std, system_noise_yaw_std;
    double system_noise_a_var, system_noise_yaw_var;

    ros::param::get("/system_noise_a_std", system_noise_a_std);
    ros::param::get("/system_noise_yaw_std", system_noise_yaw_std);

    system_noise_a_var = pow(system_noise_a_std, 2);
    system_noise_yaw_var = pow(system_noise_yaw_std, 2);

    Q(0,0) = system_noise_a_var * pow(delta_t, 6) / 36.0;
    Q(0,2) = system_noise_a_var * pow(delta_t, 5) / 12.0;
    Q(0,4) = system_noise_a_var * pow(delta_t, 4) / 6.0;
    Q(1,1) = system_noise_a_var * pow(delta_t, 6) / 36.0;
    Q(1,3) = system_noise_a_var * pow(delta_t, 5) / 12.0;
    Q(1,5) = system_noise_a_var * pow(delta_t, 4) / 6.0;
    Q(2,0) = system_noise_a_var * pow(delta_t, 5) / 12.0;
    Q(2,2) = system_noise_a_var * pow(delta_t, 4) / 4.0;
    Q(2,4) = system_noise_a_var * pow(delta_t, 3) / 2.0;
    Q(3,1) = system_noise_a_var * pow(delta_t, 5) / 12.0;
    Q(3,3) = system_noise_a_var * pow(delta_t, 4) / 4.0;
    Q(3,5) = system_noise_a_var * pow(delta_t, 3) / 2.0;
    Q(4,0) = system_noise_a_var * pow(delta_t, 4) / 6.0;
    Q(4,2) = system_noise_a_var * pow(delta_t, 3) / 2.0;
    Q(4,4) = system_noise_a_var * pow(delta_t, 2);
    Q(5,1) = system_noise_a_var * pow(delta_t, 4) / 6.0;
    Q(5,3) = system_noise_a_var * pow(delta_t, 3) / 2.0;
    Q(5,5) = system_noise_a_var * pow(delta_t, 2);

    Q(6,6) = system_noise_yaw_var * pow(delta_t, 4) / 4.0;
    Q(6,7) = system_noise_yaw_var * pow(delta_t, 3) / 2.0;
    Q(7,6) = system_noise_yaw_var * pow(delta_t, 3) / 2.0;
    Q(7,7) = system_noise_yaw_var * pow(delta_t, 2);

    P_k1_k = F*P_k*F.transpose() + Q;
}

void TrackingObject::calc_kalman_gain()
{
    R.setIdentity();
    S = H * P_k1_k * H.transpose() + R;
    K = P_k1_k * H.transpose() * S.inverse();
}

void TrackingObject::calc_st(double x, double y, double yaw)
{
    Matrix<double, _st_DIM, 1> st_temp;
    Matrix<double, _Z_DIM, 1> z;
    z(0,0) = x;
    z(1,0) = y;
    z(2,0) = yaw;
    
    st_temp = st_k1_k + K * (z - H * st_k1_k);
    st_k =  st_temp;
}

void TrackingObject::calc_P()
{
    Matrix<double, _st_DIM, _st_DIM> P_temp;
    P_temp = P_k1_k - K * H * P_k1_k;
    P_k = P_temp;
}