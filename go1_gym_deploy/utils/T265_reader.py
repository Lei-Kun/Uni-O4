import pyrealsense2.pyrealsense2 as rs
import math
import numpy as np
import time

class RealSensePose:
    def __init__(self):
        self.reset()
    def reset(self, ):
        try:
            self.pipe = rs.pipeline()
            
            cfg = rs.config()
            cfg.enable_stream(rs.stream.pose)
            self.pipe.start(cfg)
            
            print("T265 pipeline enabled")
        except:
            print("T265 start pipe fail, retry after 2 s")
            time.sleep(2.0)
            self.reset()

    def __del__(self):
        self.pipe.stop()

    # Define a function to invert/reverse the coordinate system.
    def flip(self, x, y, z, w):
        return -z, -x, y, w

    # Define a function to convert quaternions to Euler angles.
    def to_euler_angle(self, q):
        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (q[3] * q[0] + q[1] * q[2])
        cosr_cosp = 1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1])
        roll = math.atan2(sinr_cosp, cosr_cosp);

        # pitch (y-axis rotation)
        sinp = 2.0 * (q[3] * q[1] - q[2] * q[0])
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp) # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp);

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (q[3] * q[2] + q[0] * q[1])
        cosy_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])  
        yaw = math.atan2(siny_cosp, cosy_cosp);
        
        return roll, pitch, yaw

    def getCurrentPoseData(self):
        frames = self.pipe.wait_for_frames()
        pose_frame = frames.first_or_default(rs.stream.pose)

        if pose_frame is not None:
            pose_data = pose_frame.as_pose_frame().get_pose_data()
            data = [pose_data.translation.x, pose_data.translation.y, pose_data.translation.z,
                    pose_data.velocity.x, pose_data.velocity.y, pose_data.velocity.z,
                    pose_data.acceleration.x, pose_data.acceleration.y, pose_data.acceleration.z]
            return data
        else:
            raise Exception("Cannot retrieve pose frame")

    def appendPoseData(self, data=None):
        try:
            time1 = time.time()
            frames = self.pipe.wait_for_frames()
            time2 = time.time()
            pose_frame = frames.first_or_default(rs.stream.pose)
            time3 = time.time()

            if pose_frame is not None:
                pose_data = pose_frame.as_pose_frame().get_pose_data()
                x, y, z, w = self.flip(pose_data.rotation.x, pose_data.rotation.y, pose_data.rotation.z, pose_data.rotation.w)
                
                roll, pitch, yaw = self.to_euler_angle((x, y, z, w))

                pose_data_list = [-pose_data.translation.z, -pose_data.translation.x, pose_data.translation.y, 
                            -pose_data.velocity.z, -pose_data.velocity.x, pose_data.velocity.y,
                            -pose_data.acceleration.z, -pose_data.acceleration.x, pose_data.acceleration.y,
                            roll, pitch, yaw,
                            -pose_data.angular_velocity.z, -pose_data.angular_velocity.x, pose_data.angular_velocity.y,
                            -pose_data.angular_acceleration.z, -pose_data.angular_acceleration.x, pose_data.angular_acceleration.y]
                time4 = time.time()
                # print('delta, time1: {}, time2: {}, time3: {}'.format(time2-time1, time3-time2, time4-time3))
                if data is not None:
                    data.extend(pose_data_list)
                    # print('shape--------------------------', np.array(data).shape[0])
                    assert np.array(data).shape[0] == 76, 'pose_data_list: {}'.format(pose_data_list)
                return True
            else:
                print("T265 frame is none!")
                return False
        except Exception as e:
            print("T265 error! error type:{}, error str:{}".format(type(e),str(e)))
            print('error file:{}'.format(e.__traceback__.tb_frame.f_globals["__file__"]))
            print('error line:{}'.format(e.__traceback__.tb_lineno))
            return False
