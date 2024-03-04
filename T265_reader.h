#include <iostream>
#include <vector>
#include <librealsense2/rs.hpp>

class RealSensePose {
private:
    rs2::pipeline pipe;
public:
    RealSensePose() {
        std::cout<<"enable t265 pipeline"<<std::endl;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_POSE);
        pipe.start(cfg);
        std::cout<<"enabled"<<std::endl;
    }
    ~RealSensePose() {
        pipe.stop();
    }

    // 定义一个结构体来存储三维向量
    struct vec3 {
        double x, y, z;
    };

    // 定义一个结构体来存储四元数
    struct quat {
        double x, y, z, w;
    };

    // 定义函数来翻转坐标系
    vec3 flip(const vec3& v) {
        return { -v.z, -v.x, v.y };
    }

    quat flip(const quat& q) {
        return { -q.z, -q.x, q.y, q.w };
    }

    // 定义函数来转换四元数为欧拉角
    void to_euler_angle(const quat& q, double& roll, double& pitch, double& yaw)
    {
        // roll (x-axis rotation)
        double sinr_cosp = +2.0 * (q.w * q.x + q.y * q.z);
        double cosr_cosp = +1.0 - 2.0 * (q.x * q.x + q.y * q.y);
        roll = atan2(sinr_cosp, cosr_cosp);

        // pitch (y-axis rotation)
        double sinp = +2.0 * (q.w * q.y - q.z * q.x);
        if (fabs(sinp) >= 1)
            pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
        else
            pitch = asin(sinp);

        // yaw (z-axis rotation)
        double siny_cosp = +2.0 * (q.w * q.z + q.x * q.y);
        double cosy_cosp = +1.0 - 2.0 * (q.y * q.y + q.z * q.z);  
        yaw = atan2(siny_cosp, cosy_cosp);
    }

    std::vector<float> getCurrentPoseData() {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::pose_frame pose_frame = frames.first_or_default(RS2_STREAM_POSE);
        if (pose_frame)
        {
            std::vector<float> data(9);
            auto poseData = pose_frame.get_pose_data();
            data[0] = poseData.translation.x;
            data[1] = poseData.translation.y;
            data[2] = poseData.translation.z;
            data[3] = poseData.velocity.x;
            data[4] = poseData.velocity.y;
            data[5] = poseData.velocity.z;
            data[6] = poseData.acceleration.x;
            data[7] = poseData.acceleration.y;
            data[8] = poseData.acceleration.z;
            return data;
        }
        else
        {
            throw std::runtime_error("Cannot retrieve pose frame");
        }
    }

    std::vector<float> appendPoseData(std::vector<float>& data) {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::pose_frame pose_frame = frames.first_or_default(RS2_STREAM_POSE);
        if (pose_frame)
        {
            auto poseData = pose_frame.get_pose_data();

            quat q = flip({ poseData.rotation.x, poseData.rotation.y, poseData.rotation.z, poseData.rotation.w });
            double roll, pitch, yaw;
            to_euler_angle(q, roll, pitch, yaw);
            
            data.push_back(-poseData.translation.z);
            data.push_back(-poseData.translation.x);
            data.push_back(poseData.translation.y);
            data.push_back(-poseData.velocity.z);
            data.push_back(-poseData.velocity.x);
            data.push_back(poseData.velocity.y);
            data.push_back(-poseData.acceleration.z);
            data.push_back(-poseData.acceleration.x);
            data.push_back(poseData.acceleration.y);

            data.push_back(roll);
            data.push_back(pitch);
            data.push_back(yaw);
            data.push_back(-poseData.angular_velocity.z);
            data.push_back(-poseData.angular_velocity.x);
            data.push_back(poseData.angular_velocity.y);
            data.push_back(-poseData.angular_acceleration.z);
            data.push_back(-poseData.angular_acceleration.x);
            data.push_back(poseData.angular_acceleration.y);
            vector<float> euler{roll, pitch, yaw};
            
            // std::cout << "pos: "<< -poseData.translation.z<<"   "<<-poseData.translation.x<<"   "<<poseData.translation.y<<std::endl;
            // std::cout << "pos: "<< -poseData.translation.z<<"   "<<-poseData.translation.x<<"   "<<poseData.translation.y<<std::endl;
            // std::cout << "pos: "<< -poseData.translation.z<<"   "<<-poseData.translation.x<<"   "<<poseData.translation.y<<std::endl;

            return euler;
        }
        else
        {
            throw std::runtime_error("Cannot retrieve pose frame");
        }
    }

};