#ifndef STEREO_DEPTH_NODE__STEREO_DEPTH_NODE_HPP_
#define STEREO_DEPTH_NODE__STEREO_DEPTH_NODE_HPP_

#include "stereo_depth/stereo_depth.hpp"

#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <pcl_ros/transforms.hpp>
#include <spdlog/spdlog.h>

#include <memory>
#include <string>

class StereoDepthNode : public rclcpp::Node
{
public:
  StereoDepthNode();
  ~StereoDepthNode();

private:
  // Params
  std::string left_img_topic_;
  std::string right_img_topic_;
  std::string camera_frame_, base_frame_;
  std::string depth_weight_file_;
  CAMParams cam_params_;

  // Variables
  cv::Mat init_left_img_, init_right_img_;
  std::unique_ptr<StereoDepthEstimation> stereodepth_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  // Subscribers
  message_filters::Subscriber<sensor_msgs::msg::Image> left_img_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> right_img_sub_;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                          sensor_msgs::msg::Image>
    SyncPolicy;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Publishers
  image_transport::Publisher depth_img_pub_;
  image_transport::Publisher sgm_disparity_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr depth_cloud_pub_;

  // Callbacks
  void stereoCallback(const sensor_msgs::msg::Image::ConstSharedPtr &left,
                      const sensor_msgs::msg::Image::ConstSharedPtr &right);
  void timerCallback();

  void
  publishImage(const image_transport::Publisher &, const cv::Mat &, const std::string &);
  void
  publishDepthCloud(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &);
  sensor_msgs::msg::PointCloud2
  transformPointCloud(const sensor_msgs::msg::PointCloud2 &, const std::string &,
                      const std::string &);
};

#endif // STEREO_DEPTH_NODE__STEREO_DEPTH_NODE_HPP_
