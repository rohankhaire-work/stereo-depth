#ifndef STEREO_DEPTH_ESTIMATION__STEREO_DEPTH_ESTIMATION_HPP_
#define STEREO_DEPTH_ESTIMATION__STEREO_DEPTH_ESTIMATION_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct CAMParams
{
  int orig_h;
  int orig_w;
  int network_h;
  int network_w;
  int network_c;
  double fx;
  double fy;
  double cx;
  double cy;
  double baseline;
};

class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char *msg) noexcept override
  {
    if(severity <= Severity::kINFO)
      std::cout << "[TRT] " << msg << std::endl;
  }
};

class StereoDepthEstimation
{
public:
  StereoDepthEstimation(const CAMParams &, const std::string &);
  ~StereoDepthEstimation();

  void runInference(const cv::Mat &, const cv::Mat &);

  cv::Mat depth_img_;
  cv::Mat depth_map_;
  sensor_msgs::msg::PointCloud2 depth_cloud_;

private:
  int resize_h_, resize_w_, channels_;
  Logger gLogger_;
  std::vector<float> result_;
  double fx_, fy_, cx_, cy_, baseline_;
  CAMParams cam_params_;
  bool use_rgb_ = true;
  float scaled_fx_, scaled_fy_, scaled_cx_, scaled_cy_;
  const double MAX_DEPTH = 80.0;
  const double MIN_DEPTH = 0.5;

  // Buffers
  void *buffers_[4];
  float *input_left_host_ = nullptr;
  float *input_right_host_ = nullptr;
  float *output_host_ = nullptr;

  // Tensorrt
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream_;

  std::vector<float> preprocessRGB(const cv::Mat &);
  std::vector<float> imageToTensor(const cv::Mat &);
  cv::Mat normalizeRGB(const cv::Mat &input);
  void initializeTRT(const std::string &);
  void computeDepthMap(std::vector<float> &);
  void convertToDepthImg(const cv::Mat &, cv::Mat &);
  void initializeDepthCloud();
  void createPointCloudFromDepth(const cv::Mat &, const cv::Mat &);
};

#endif // STEREO_DEPTH_ESTIMATION_STEREO_DEPTH_ESTIMATION_HPP_
