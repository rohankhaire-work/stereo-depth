#include "stereo_depth/stereo_depth_node.hpp"
#include <rclcpp/logging.hpp>

StereoDepthNode::StereoDepthNode() : Node("stereo_depth_node")
{
  // Set parameters
  left_img_topic_ = declare_parameter<std::string>("left_image_topic", "");
  right_img_topic_ = declare_parameter<std::string>("right_image_topic", "");
  camera_frame_ = declare_parameter<std::string>("camera_frame", "");
  base_frame_ = declare_parameter<std::string>("base_frame", "");
  cam_params_.network_h = declare_parameter("input_network_height", 192);
  cam_params_.network_w = declare_parameter("input_network_width", 640);
  cam_params_.network_c = declare_parameter("input_network_channel", 3);
  cam_params_.orig_h = declare_parameter("original_iamge_height", 374);
  cam_params_.orig_w = declare_parameter("original_image_width", 1230);
  depth_weight_file_ = declare_parameter<std::string>("depth_weights_file", "");
  cam_params_.fx = declare_parameter("fx", 0.0);
  cam_params_.fy = declare_parameter("fy", 0.0);
  cam_params_.cx = declare_parameter("cx", 0.0);
  cam_params_.cy = declare_parameter("cy", 0.0);
  cam_params_.baseline = declare_parameter("baseline", 0.0);

  if(left_img_topic_.empty() || right_img_topic_.empty() || depth_weight_file_.empty())
  {
    RCLCPP_ERROR(get_logger(),
                 "Check if topic name or weight file is assigned in params");
    return;
  }

  // Message filter subcriber
  left_img_sub_.subscribe(this, left_img_topic_);
  right_img_sub_.subscribe(this, right_img_topic_);
  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
    SyncPolicy(10), left_img_sub_, right_img_sub_);
  sync_->registerCallback(std::bind(&StereoDepthNode::stereoCallback, this,
                                    std::placeholders::_1, std::placeholders::_2));

  timer_ = this->create_wall_timer(std::chrono::milliseconds(50),
                                   std::bind(&StereoDepthNode::timerCallback, this));

  depth_img_viz_pub_ = image_transport::create_publisher(this, "/stereo_depth_viz");
  depth_img_pub_ = image_transport::create_publisher(this, "/stereo_depth");
  depth_cloud_pub_
    = create_publisher<sensor_msgs::msg::PointCloud2>("/stereo_depth_cloud", 10);

  // Get weight paths
  std::string share_dir
    = ament_index_cpp::get_package_share_directory("stereo_depth_estimation");
  std::string depth_weight_path = share_dir + depth_weight_file_;

  // Initialize TensorRT
  stereodepth_ = std::make_unique<StereoDepthEstimation>(cam_params_, depth_weight_path);

  // Initialize tf2 for transforms
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
}

StereoDepthNode::~StereoDepthNode()
{
  timer_->cancel();
  stereodepth_.reset();
}

void StereoDepthNode::stereoCallback(const sensor_msgs::msg::Image::ConstSharedPtr &left,
                                     const sensor_msgs::msg::Image::ConstSharedPtr &right)
{
  // Convert ROS2 image message to OpenCV format
  try
  {
    cv_bridge::CvImagePtr init_left_ptr = cv_bridge::toCvCopy(left, "rgb8");
    cv_bridge::CvImagePtr init_right_ptr = cv_bridge::toCvCopy(right, "rgb8");

    // Check if the ptr is present
    if(!init_left_ptr || !init_right_ptr)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge::toCvCopy() returned nullptr!");
      return;
    }

    // Copy the image
    init_left_img_ = init_left_ptr->image;
    init_right_img_ = init_right_ptr->image;
  }
  catch(cv_bridge::Exception &e)
  {
    RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
    return;
  }
}

void StereoDepthNode::timerCallback()
{
  // Check if the image and pointcloud exists
  if(init_left_img_.empty() || init_right_img_.empty())
  {
    RCLCPP_WARN(this->get_logger(), "Image is missing in StereoDepthNode");
    return;
  }

  auto start_time = std::chrono::steady_clock::now();
  // Run Stereo depth estimation
  // Calcualted Time is for NN computation
  // and resizing
  stereodepth_->runInference(init_left_img_, init_right_img_);

  auto end_time = std::chrono::steady_clock::now();
  auto duration_ms
    = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  RCLCPP_INFO(this->get_logger(), "Inference took %ld ms", duration_ms);

  // Publsuh depth image and depth cloud
  publishImage(depth_img_viz_pub_, stereodepth_->depth_img_, "rgb8");
  publishImage(depth_img_pub_, stereodepth_->depth_map_, "32FC1");
  publishDepthCloud(depth_cloud_pub_);
}

void StereoDepthNode::publishImage(const image_transport::Publisher &pub,
                                   const cv::Mat &image, const std::string &encoding)
{
  cv::Mat final_img = image;
  // Convert OpenCV image to ROS2 message
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  sensor_msgs::msg::Image::SharedPtr msg
    = cv_bridge::CvImage(header, encoding, final_img).toImageMsg();

  // Publish image
  pub.publish(*msg);
}

void StereoDepthNode::publishDepthCloud(
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub)
{
  // Fill the header
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  header.frame_id = camera_frame_;

  // Set header field
  stereodepth_->depth_cloud_.header = header;

  // Transform cloud to base link
  sensor_msgs::msg::PointCloud2 base_cloud
    = transformPointCloud(stereodepth_->depth_cloud_, camera_frame_, base_frame_);

  // Publish depth cloud
  pub->publish(base_cloud);
}

sensor_msgs::msg::PointCloud2
StereoDepthNode::transformPointCloud(const sensor_msgs::msg::PointCloud2 &input_cloud,
                                     const std::string &input_frame,
                                     const std::string &target_frame)
{
  sensor_msgs::msg::PointCloud2 transformed_cloud;
  // Lookup transform from LiDAR to Camera frame
  geometry_msgs::msg::TransformStamped transform_stamped;
  try
  {
    transform_stamped
      = tf_buffer_->lookupTransform(target_frame, input_frame, tf2::TimePointZero);
  }
  catch(tf2::TransformException &ex)
  {
    RCLCPP_ERROR(get_logger(), "Could not transform %s to %s: %s", input_frame.c_str(),
                 target_frame.c_str(), ex.what());
    return transformed_cloud;
  }

  // Apply transformation to point cloud
  pcl_ros::transformPointCloud(target_frame, transform_stamped, input_cloud,
                               transformed_cloud);

  return transformed_cloud;
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StereoDepthNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
