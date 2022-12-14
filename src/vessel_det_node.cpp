#include <ros/ros.h>
#include <nodelet/loader.h>


int main(int argc, char **argv){
  ros::init(argc, argv, "vessel_det_node");

  nodelet::Loader nodelet;
  nodelet::M_string remap(ros::names::getRemappings());
  nodelet::V_string nargv;
  
  nodelet.load(ros::this_node::getName(), "mbz_vessel_det/DetectionNodelet", remap, nargv);
  
  ros::spin();
  return 0;
}