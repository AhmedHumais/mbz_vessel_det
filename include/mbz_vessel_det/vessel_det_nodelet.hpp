#ifndef __VESSEL_DET_NODELET_H__
#define __VESSEL_DET_NODELET_H__

#include <cstdio>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "ros/ros.h"
#include "nodelet/nodelet.h"
#include "sensor_msgs/Image.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Bool.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/PoseStamped.h"
#include <image_transport/image_transport.h>
#include "cv_bridge/cv_bridge.h"

#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2/LinearMath/Transform.h"
// #include "tf2/convert.h"
//#include "/home/vision/mbzirc_ws/src/ros_ign/ros_ign_bridge/include/ros_ign_bridge/convert/ros_ign_interfaces.hpp"
#include "yolo_v5/yolov5.h"
#include "yolo_v5/yoloparam.h"

using namespace std;

class DetectionNodelet : public nodelet::Nodelet
{

public:
    DetectionNodelet();
    ~DetectionNodelet(){};

private:
    double H_FOV = 1.5708; // 90 deg
    void loadParam();
    virtual void onInit();
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void depthCallback(const sensor_msgs::ImageConstPtr& msg);

    ros::NodeHandle *nh_;
    ros::NodeHandle *pnh_;
    ros::Publisher target_err_pub;
    ros::Publisher inertial_cord_pub;
    ros::Publisher detection_pub;
    ros::Subscriber depth_sub;
    ros::Subscriber image_sub;

    image_transport::Publisher det_img_pub;

    tf2_ros::Buffer tf_Buffer;

    string image_sub_topic_name;
    string depth_sub_topic_name;
    string namewindow;
    string video_saving_path;
    Yolov5 *yolov5;
    cv::Rect target_bb;
    cv::Point target_center;
    double MAX_depth_val = 10;
    bool show_screen;

public:
    yoloparam YP;
    void* YParam;
    
    int target_cnt;
    int not_target_cnt;
    float angle;
    int status;
    bool detected = false;
    int saving_cnt;
    std::string cat_need_todet;
};

#endif // __VESSEL_DET_NODELET_H__