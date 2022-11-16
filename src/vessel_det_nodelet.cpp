#include "mbz_vessel_det/vessel_det_nodelet.hpp"
#include <cmath>
#include <iostream>

#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(DetectionNodelet, nodelet::Nodelet)

DetectionNodelet::DetectionNodelet()
{
}

void DetectionNodelet::loadParam()
{

  pnh_->param<string>("image_sub_topic_name", image_sub_topic_name, "/zed_node/rgb/image_rect_color");
  pnh_->param<string>("depth_sub_topic_name", depth_sub_topic_name, "/zed_node/depth/depth_registered");
  pnh_->param<string>("namewindow", namewindow, "vessel_det");
  pnh_->param<int>("device", YP.DEVICE, 0);
  pnh_->param<float>("nms", YP.NMS_THRESH, 0.5);
  pnh_->param<float>("conf", YP.CONF_THRESH, 0.7);
  pnh_->param<int>("batch_size", YP.BATCH_SIZE, 1);
  pnh_->param<int>("input_h", YP.INPUT_H, 640);
  pnh_->param<int>("input_w", YP.INPUT_W, 640);
  pnh_->param<int>("class_num", YP.CLASS_NUM, 2);
  pnh_->param<string>("engine_dir", YP.ENGINE_DIR, "engine/vessel.engine");
  pnh_->param<double>("MAX_DEPTH", MAX_depth_val, 15);
  pnh_->param<bool>("show_det_output", show_screen, false);

  ROS_INFO("class_num: '%d'", YP.CLASS_NUM);
  YParam = &YP;
}

void DetectionNodelet::onInit()
{
  nh_ = &(getNodeHandle());
  pnh_ = &(getPrivateNodeHandle());
  this->loadParam();
  target_cnt = 0;
  not_target_cnt = 0;
  float angle;
  yolov5 = new Yolov5(YParam);
  status = 1;
  cat_need_todet = "vessel_a";

  tf2_ros::TransformListener tf_listener(tf_Buffer);

  image_transport::ImageTransport it(*nh_);
  det_img_pub = it.advertise("/detection/image", 2);

  target_err_pub = nh_->advertise<std_msgs::Float32>("/usv/target/error", 1000);
  inertial_cord_pub = nh_->advertise<geometry_msgs::Point>("/usv/target/cord", 1000);
  detection_pub = nh_->advertise<std_msgs::Bool>("/usv/target/is_detected", 1000);

  depth_sub = nh_->subscribe<sensor_msgs::Image>(depth_sub_topic_name, 3, &DetectionNodelet::depthCallback, this);

  image_sub = nh_->subscribe<sensor_msgs::Image>(image_sub_topic_name, 3, &DetectionNodelet::imageCallback, this);
}

void DetectionNodelet::depthCallback(const sensor_msgs::ImageConstPtr &msg)
{
  if (detected)
  {
    detected = false;
    cv_bridge::CvImagePtr cv_ptr_img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);

    std::cout << target_bb.x << ", " << target_bb.y << ", " << target_bb.width << ", " << target_bb.height << std::endl;

    auto depth_img = cv_ptr_img->image;

    if (target_bb.x < 1 || target_bb.y < 1 || (target_bb.x + target_bb.width) >= depth_img.cols || (target_bb.y + target_bb.height) >= depth_img.rows)
    {
      return;
    }

    auto cropped_img = depth_img(target_bb);
    double min_depth;
    cv::minMaxLoc(cropped_img, &min_depth);

    vector<float> array;
    if (cropped_img.isContinuous())
    {
      // array.assign((float*)mat.datastart, (float*)mat.dataend); // <- has problems for sub-matrix like mat = big_mat.row(i)
      array.assign((float *)cropped_img.data, (float *)cropped_img.data + cropped_img.total() * cropped_img.channels());
    }
    else
    {
      for (int i = 0; i < cropped_img.rows; ++i)
      {
        array.insert(array.end(), cropped_img.ptr<float>(i), cropped_img.ptr<float>(i) + cropped_img.cols * cropped_img.channels());
      }
    }
    vector<float> depths;
    for (int i = 0; i < array.size(); i++)
    {
      if (array[i] != INFINITY && array[i] != NAN)
      {
        depths.push_back(array[i]);
      }
    }
    if (depths.empty())
    {
      return;
    }
    std::sort(depths.begin(), depths.end());
    min_depth = depths[depths.size() / 2];

    // std::cout << min_depth << std::endl;
    if (min_depth > 0.5 && min_depth < MAX_depth_val)
    {
      auto from_frame = msg->header.frame_id;
      std::string to_frame = "usv";

      geometry_msgs::TransformStamped transformStamped;

      // Look up for the transformation between target_frame and turtle2 frames
      // and send velocity commands for turtle2 to reach target_frame
      
      // std::cout << transformStamped.transform.translation.x <<std::endl;

      geometry_msgs::PoseStamped tgt_loc;
      float f_ = 554.3;
      tgt_loc.pose.position.y = -(target_center.x / f_) * min_depth;
      tgt_loc.pose.position.z = -(target_center.y / f_) * min_depth;
      tgt_loc.pose.position.x = min_depth;
      tgt_loc.header = msg->header;

      try
      {
        tgt_loc = tf_Buffer.transform(tgt_loc, to_frame, ros::Duration(0.1));
        // transformStamped = tf_Buffer.lookupTransform(to_frame, from_frame, ros::Time(0));
      }
      catch (tf2::TransformException &ex)
      {
        ROS_WARN("Transform to usv frame not provided (will use identity). Please provide.");
      }
      // tf2::doTransform(tgt_loc, tgt_loc, transformStamped);

      inertial_cord_pub.publish(tgt_loc.pose.position);
    }
    std::cout << "--" << std::endl;
  }
}

void DetectionNodelet::imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
  cv_bridge::CvImagePtr cv_ptr_img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  // cvtColor(cv_ptr_img->image, cv_ptr_img->image, COLOR_BGR2RGB);
  Mat out_img;
  vector<_bbox> bbox;
  yolov5->objDetection(cv_ptr_img->image, out_img, bbox); // Box

  bool target_flag = false;

  for (auto obj : bbox)
  {
    if (obj.class_id == cat_need_todet)
    {
      target_flag = true;
      if (target_cnt >= 5)
      {
        target_bb.x = obj.xmin;
        target_bb.y = obj.ymin;
        target_bb.width = abs(obj.xmax - obj.xmin);
        target_bb.height = abs(obj.ymax - obj.ymin);
        auto tgt_center_x = (obj.xmin + obj.xmax) / 2.0;
        auto img_center_x = obj.img_width / 2.0;
        auto err_x = img_center_x - tgt_center_x;

        auto ref_ang = H_FOV / 2.0;
        std_msgs::Float32 err_msg;
        err_msg.data = (err_x / img_center_x) * ref_ang;

        target_err_pub.publish(err_msg);

        target_center.x = tgt_center_x - img_center_x;
        target_center.y = (obj.ymin + obj.ymax) / 2.0 - obj.img_height / 2.0;
        detected = true;
      }
    }
  }
  if (target_flag)
  {
    target_cnt++;
  }
  else
  {
    target_cnt = 0;
  }
  std_msgs::Bool is_detected_msg;
  is_detected_msg.data = target_flag;
  detection_pub.publish(is_detected_msg);
  sensor_msgs::ImagePtr det_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", out_img).toImageMsg();
  det_img_pub.publish(det_img_msg);
  
  if (show_screen)
  {
    cv::namedWindow(namewindow, 0);
    cv::resizeWindow(namewindow, 640, 480);
    cv::imshow(namewindow, out_img);
    cv::waitKey(5);
  }
}
