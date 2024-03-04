#include <cmath>
#include <iostream>
#include <vector>


std::vector<float> compute_gravity_vector(float roll, float pitch, float yaw) {
    // Convert euler angles to rotation matrix
    float cos_roll = cos(roll);
    float sin_roll = sin(roll);
    float cos_pitch = cos(pitch);
    float sin_pitch = sin(pitch);
    float cos_yaw = cos(yaw);
    float sin_yaw = sin(yaw);

    std::vector<std::vector<float>> rotation_matrix = {
        {cos_yaw*cos_pitch, cos_yaw*sin_pitch*sin_roll - sin_yaw*cos_roll, cos_yaw*sin_pitch*cos_roll + sin_yaw*sin_roll},
        {sin_yaw*cos_pitch, sin_yaw*sin_pitch*sin_roll + cos_yaw*cos_roll, sin_yaw*sin_pitch*cos_roll - cos_yaw*sin_roll},
        {-sin_pitch, cos_pitch*sin_roll, cos_pitch*cos_roll}
    };

    // Gravity vector in world frame
    std::vector<float> gravity_world = {0, 0, -9.81};
    std::vector<float> gravity_unit = {0, 0, -1.0};
    // Compute gravity vector in robot frame
    std::vector<float> gravity_robot(3, 0);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            gravity_robot[i] += rotation_matrix[i][j] * gravity_unit[j];
        }
    }

    return gravity_robot;
}