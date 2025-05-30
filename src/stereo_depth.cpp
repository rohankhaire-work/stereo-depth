#include "stereo_depth/stereo_depth.hpp"

StereoDepthEstimation::StereoDepthEstimation(const CAMParams &cam_params,
                                             const std::string &weight_file)
{
  // Set CAM and SGM params
  cam_params_ = cam_params;

  resize_h_ = cam_params_.network_h;
  resize_w_ = cam_params_.network_w;
  channels_ = cam_params_.network_c;

  // Scale camera intrinsics
  float scale_x = static_cast<float>(cam_params.network_w) / cam_params_.orig_w;
  float scale_y = static_cast<float>(cam_params.network_h) / cam_params_.orig_h;

  scaled_fx_ = static_cast<float>(cam_params_.fx) * scale_x;
  scaled_fy_ = static_cast<float>(cam_params_.fy) * scale_y;
  scaled_cx_ = static_cast<float>(cam_params_.cx) * scale_x;
  scaled_cy_ = static_cast<float>(cam_params_.cy) * scale_y;

  // Initialize TRT
  initializeTRT(weight_file);

  // Allocate buffers
  cudaMallocHost(reinterpret_cast<void **>(&input_left_host_),
                 1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMallocHost(reinterpret_cast<void **>(&input_right_host_),
                 1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers_[0], 1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers_[1], 1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMallocHost(reinterpret_cast<void **>(&output_host_),
                 resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers_[2], resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers_[3], resize_h_ * resize_w_ * sizeof(float));

  // Create CUDA stream
  cudaStreamCreate(&stream_);

  // Resize depth map
  depth_map_.resize(resize_h_ * resize_w_);

  // initialize depth cloud
  initializeDepthCloud();
}

StereoDepthEstimation::~StereoDepthEstimation()
{
  if(buffers_[0])
  {
    cudaFree(buffers_[0]);
    buffers_[0] = nullptr;
  }
  if(buffers_[1])
  {
    cudaFree(buffers_[1]);
    buffers_[1] = nullptr;
  }
  if(buffers_[2])
  {
    cudaFree(buffers_[2]);
    buffers_[2] = nullptr;
  }
  if(buffers_[3])
  {
    cudaFree(buffers_[3]);
    buffers_[3] = nullptr;
  }
  if(input_left_host_)
  {
    cudaFreeHost(input_left_host_);
    input_left_host_ = nullptr;
  }
  if(input_right_host_)
  {
    cudaFreeHost(input_right_host_);
    input_right_host_ = nullptr;
  }
  if(output_host_)
  {
    cudaFreeHost(output_host_);
    output_host_ = nullptr;
  }
  if(stream_)
  {
    cudaStreamDestroy(stream_);
  }
}

cv::Mat StereoDepthEstimation::normalizeRGB(const cv::Mat &input)
{
  std::vector<cv::Mat> channels(3);
  cv::split(input, channels);

  std::vector<cv::Mat> temp_data;
  temp_data.resize(3);

  for(int i = 0; i < 3; ++i)
  {
    cv::Mat float_channel;
    channels[i].convertTo(float_channel, CV_32F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(float_channel, mean, stddev);

    // Normalize: (x - mean) / std
    temp_data[i] = (float_channel - mean[0]) / stddev[0];
  }

  // Convert to cv::Mat
  cv::Mat normalized_rgb;
  cv::vconcat(temp_data, normalized_rgb);

  return normalized_rgb;
}

std::vector<float> StereoDepthEstimation::preprocessRGB(const cv::Mat &image)
{
  cv::Mat resized, float_image;

  // Resize to model input size
  cv::resize(image, resized, cv::Size(resize_w_, resize_h_));

  // Convert to float32 and CHW order
  float_image = normalizeRGB(resized);

  // Convert to Tensor
  std::vector<float> img_tensor = imageToTensor(float_image);

  return img_tensor;
}

std::vector<float> StereoDepthEstimation::imageToTensor(const cv::Mat &mat)
{
  std::vector<float> tensor_data;
  if(mat.isContinuous())
    tensor_data.assign((float *)mat.datastart, (float *)mat.dataend);
  else
  {
    // Convert from HWC to CHW
    if(mat.channels() == 1)
    {
      // Single-channel (grayscale)
      for(int i = 0; i < mat.rows; ++i)
      {
        const float *row_ptr = mat.ptr<float>(i);
        tensor_data.insert(tensor_data.end(), row_ptr, row_ptr + mat.cols);
      }
    }
    else
    {
      // Multi-channel (e.g., RGB = 3 channels)
      for(int c = 0; c < mat.channels(); ++c)
      {
        for(int i = 0; i < mat.rows; ++i)
        {
          for(int j = 0; j < mat.cols; ++j)
          {
            const cv::Vec<float, 3> &pixel = mat.at<cv::Vec<float, 3>>(i, j);
            tensor_data.push_back(pixel[c]);
          }
        }
      }
    }
  }
  return tensor_data;
}

void StereoDepthEstimation::initializeTRT(const std::string &engine_file)
{
  // Load TensorRT engine from file
  std::ifstream file(engine_file, std::ios::binary);
  if(!file)
  {
    throw std::runtime_error("Failed to open engine file: " + engine_file);
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);

  // Create runtime and deserialize engine
  // Create TensorRT Runtime
  runtime.reset(nvinfer1::createInferRuntime(gLogger_));

  // Deserialize engine
  engine.reset(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  context.reset(engine->createExecutionContext());
}

void StereoDepthEstimation::runInference(const cv::Mat &left, const cv::Mat &right)
{
  // Preprocess Stereo images and convert to Tensor
  std::vector<float> left_img_tensor = preprocessRGB(left);
  std::vector<float> right_img_tensor = preprocessRGB(right);

  // Copy to host memory and then to GPU
  std::memcpy(input_left_host_, left_img_tensor.data(),
              1 * channels_ * resize_h_ * resize_w_ * sizeof(float));
  std::memcpy(input_left_host_, right_img_tensor.data(),
              1 * channels_ * resize_h_ * resize_w_ * sizeof(float));

  cudaMemcpyAsync(buffers_[0], input_left_host_,
                  1 * channels_ * resize_h_ * resize_w_ * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(buffers_[1], input_right_host_,
                  1 * channels_ * resize_h_ * resize_w_ * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);

  // Set up inference buffers
  context->setInputTensorAddress("left", buffers_[0]);
  context->setInputTensorAddress("right", buffers_[1]);
  context->setOutputTensorAddress("output", buffers_[2]);
  context->setOutputTensorAddress("disp", buffers_[3]);

  // inference
  context->enqueueV3(stream_);

  // Copy the result back
  cudaMemcpyAsync(output_host_, buffers_[2], resize_h_ * resize_w_ * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);

  cudaStreamSynchronize(stream_);

  int output_size = resize_h_ * resize_w_;
  result_.assign(output_host_, output_host_ + output_size);

  // Compute depth map from inference disparity
  computeDepthMap(depth_map_);

  // Convert to cv::Mat
  cv::Mat depth_map(resize_h_, resize_w_, CV_32FC1, depth_map_.data());

  // Compute depth image
  convertToDepthImg(depth_map, depth_img_);

  // convert to depth cloud
  cv::Mat resized_img;
  cv::resize(left, resized_img, cv::Size(resize_w_, resize_h_));
  createPointCloudFromDepth(depth_map, resized_img);
}

void StereoDepthEstimation::computeDepthMap(std::vector<float> &depth_map)
{
  double epsilon = 1e-6;

  for(int i = 0; i < result_.size(); ++i)
  {
    // Re-adjust disparity
    float disp = result_[i];
    if(disp > epsilon)
    {
      depth_map[i] = (scaled_fx_ * cam_params_.baseline) / disp;
    }
    else
    {
      depth_map[i] = 0.0f;
    }
  }
}

void StereoDepthEstimation::convertToDepthImg(const cv::Mat &depth_map,
                                              cv::Mat &depth_img)
{
  cv::Mat depth_vis;
  depth_map.convertTo(depth_img, CV_8U);
  cv::applyColorMap(depth_img, depth_img, cv::COLORMAP_JET);
}

void StereoDepthEstimation::initializeDepthCloud()
{
  // Fill the pcd infomation based on the image
  depth_cloud_.height = resize_h_;
  depth_cloud_.width = resize_w_;
  depth_cloud_.is_bigendian = false;
  depth_cloud_.is_dense = false;

  // Define fields
  if(use_rgb_)
  {
    sensor_msgs::msg::PointField field_x;
    field_x.name = "x";
    field_x.offset = 0;
    field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_x.count = 1;

    sensor_msgs::msg::PointField field_y;
    field_y.name = "y";
    field_y.offset = 4;
    field_y.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_y.count = 1;

    sensor_msgs::msg::PointField field_z;
    field_z.name = "z";
    field_z.offset = 8;
    field_z.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_z.count = 1;

    sensor_msgs::msg::PointField field_rgb;
    field_rgb.name = "rgb";
    field_rgb.offset = 12;
    field_rgb.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_rgb.count = 1;

    depth_cloud_.fields.push_back(field_x);
    depth_cloud_.fields.push_back(field_y);
    depth_cloud_.fields.push_back(field_z);
    depth_cloud_.fields.push_back(field_rgb);
    depth_cloud_.point_step = 16;
  }
  else
  {
    sensor_msgs::msg::PointField field_x;
    field_x.name = "x";
    field_x.offset = 0;
    field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_x.count = 1;

    sensor_msgs::msg::PointField field_y;
    field_y.name = "y";
    field_y.offset = 4;
    field_y.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_y.count = 1;

    sensor_msgs::msg::PointField field_z;
    field_z.name = "z";
    field_z.offset = 8;
    field_z.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_z.count = 1;

    depth_cloud_.fields.push_back(field_x);
    depth_cloud_.fields.push_back(field_y);
    depth_cloud_.fields.push_back(field_z);
    depth_cloud_.point_step = 12;
  }

  depth_cloud_.row_step = depth_cloud_.point_step * depth_cloud_.width;
  depth_cloud_.data.resize(depth_cloud_.row_step * depth_cloud_.height);
}

void StereoDepthEstimation::createPointCloudFromDepth(const cv::Mat &depth,
                                                      const cv::Mat &rgb)
{
  sensor_msgs::PointCloud2Iterator<float> iter_x(depth_cloud_, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(depth_cloud_, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(depth_cloud_, "z");
  sensor_msgs::PointCloud2Iterator<float> iter_rgb(depth_cloud_, "rgb");

  for(int v = 0; v < depth.rows; ++v)
  {
    for(int u = 0; u < depth.cols; ++u, ++iter_x, ++iter_y, ++iter_z)
    {
      float z = depth.at<float>(v, u);

      *iter_x = (u - scaled_cx_) * z / scaled_fx_;
      *iter_y = (v - scaled_cy_) * z / scaled_fy_;
      *iter_z = z;

      if(use_rgb_)
      {
        const cv::Vec3b &color = rgb.at<cv::Vec3b>(v, u);
        uint32_t rgb_packed = (color[2] << 16) | (color[1] << 8) | (color[0]);
        float rgb_float;
        std::memcpy(&rgb_float, &rgb_packed, sizeof(float));
        *iter_rgb = rgb_float;
        ++iter_rgb;
      }
    }
  }
}
