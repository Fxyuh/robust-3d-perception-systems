#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>
#include "bevfusion/bevfusion.hpp"

namespace {
struct PacketHeader {
  char magic[4];
  uint32_t version;
  uint32_t sample_idx;
  uint32_t num_points;
  uint32_t image_bytes;
  uint32_t point_bytes;
  uint32_t matrix_bytes;
  uint32_t reserved;
};
static_assert(sizeof(PacketHeader)==32);
constexpr size_t kImageBytes=6ul*3*256*704*sizeof(uint16_t);
constexpr size_t kMatricesBytes=4ul*6*4*4*sizeof(float);
void cuda_check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
std::shared_ptr<bevfusion::Core> make_core(const std::string& m){
  bevfusion::CoreParameter p;
  p.camera_model=m+"/camera.engine";
  p.camera_vtransform=m+"/camera_vtransform.engine";
  p.transfusion=m+"/fuser.engine";
  p.transbbox.model=m+"/head_bbox.engine";
  p.transbbox.confidence_threshold=0.0f;
  p.transbbox.sorted_bboxes=true;
  p.transbbox.out_size_factor=8;
  p.transbbox.voxel_size={0.075,0.075};
  p.transbbox.pc_range={-54.0,-54.0};
  p.transbbox.post_center_range_start={-61.2,-61.2,-10.0};
  p.transbbox.post_center_range_end={61.2,61.2,10.0};
  auto& v=p.lidar_scn.voxelization;
  v.min_range={-54.0,-54.0,-5.0};
  v.max_range={54.0,54.0,3.0};
  v.voxel_size={0.075,0.075,0.2};
  v.grid_size=v.compute_grid_size(v.max_range,v.min_range,v.voxel_size);
  v.max_points_per_voxel=10;v.max_points=300000;v.max_voxels=160000;v.num_feature=5;
  p.lidar_scn.model=m+"/lidar_sparse.onnx";
  p.lidar_scn.order=bevfusion::lidar::CoordinateOrder::XYZ;
  p.lidar_scn.precision=bevfusion::lidar::Precision::Float16;
  auto& g=p.geometry;
  g.xbound={-54.0,54.0,0.3};g.ybound={-54.0,54.0,0.3};
  g.zbound={-10.0,10.0,20.0};g.dbound={1.0,60.0,0.5};
  g.image_width=704;g.image_height=256;g.feat_width=88;g.feat_height=32;g.num_camera=6;g.geometry_dim={360,360,80};
  auto& n=p.normalize;
  n.image_width=1600;n.image_height=900;n.output_width=704;n.output_height=256;n.num_camera=6;n.resize_lim=0.48f;
  float mean[3]={0.485,0.456,0.406},stdv[3]={0.229,0.224,0.225};
  n.method=bevfusion::camera::NormMethod::mean_std(mean,stdv,1/255.0f,0.0f);
  auto core=bevfusion::create_core(p);
  if(!core)throw std::runtime_error("native BEVFusion core creation failed");
  core->set_timer(false);
  return core;
}
}
class NativeNode final:public rclcpp::Node{
 public:
  NativeNode():Node("bevfusion_native"){
    auto model_dir=declare_parameter<std::string>("model_dir","");
    if(model_dir.empty())throw std::runtime_error("Pass ROS parameter model_dir with exported model files");
    core_=make_core(model_dir);
    cuda_check(cudaStreamCreate(&stream_));
    cuda_check(cudaMalloc(&image_device_,kImageBytes));
    pub_=create_publisher<std_msgs::msg::String>("/bevfusion/native_detections_json",1);
    sub_=create_subscription<std_msgs::msg::UInt8MultiArray>("/bevfusion/native_input",1,
      [this](std_msgs::msg::UInt8MultiArray::SharedPtr msg){on_packet(*msg);});
    RCLCPP_INFO(get_logger(),"Native BEVFusion ready: /bevfusion/native_input -> /bevfusion/native_detections_json");
  }
  ~NativeNode()override{
    if(image_device_)cudaFree(image_device_);
    if(stream_)cudaStreamDestroy(stream_);
  }
 private:
  void on_packet(const std_msgs::msg::UInt8MultiArray& msg){
    try{
      const auto& data=msg.data;
      if(data.size()<sizeof(PacketHeader))throw std::runtime_error("packet too short");
      PacketHeader h;std::memcpy(&h,data.data(),sizeof(h));
      if(std::memcmp(h.magic,"BFN1",4)!=0||h.version!=1)throw std::runtime_error("invalid packet version");
      if(h.num_points==0||h.num_points>300000||h.image_bytes!=kImageBytes||
         h.point_bytes!=h.num_points*5*sizeof(uint16_t)||h.matrix_bytes!=kMatricesBytes)
         throw std::runtime_error("invalid packet shapes");
      if(data.size()!=sizeof(h)+h.image_bytes+h.point_bytes+h.matrix_bytes)
         throw std::runtime_error("packet length mismatch");
      const uint8_t* image=data.data()+sizeof(h);
      const uint8_t* points=image+h.image_bytes;
      const uint8_t* matrices=points+h.point_bytes;
      std::vector<uint16_t> point_storage(h.num_points*5);
      std::memcpy(point_storage.data(),points,h.point_bytes);
      std::array<float,96> c2l,intr,l2i,aug;
      std::memcpy(c2l.data(),matrices,384);
      std::memcpy(intr.data(),matrices+384,384);
      std::memcpy(l2i.data(),matrices+768,384);
      std::memcpy(aug.data(),matrices+1152,384);
      auto t0=std::chrono::steady_clock::now();
      core_->update(c2l.data(),intr.data(),l2i.data(),aug.data(),stream_);
      cuda_check(cudaMemcpyAsync(image_device_,image,kImageBytes,cudaMemcpyHostToDevice,stream_));
      auto boxes=core_->forward_no_normalize(reinterpret_cast<const nvtype::half*>(image_device_),
        reinterpret_cast<const nvtype::half*>(point_storage.data()),h.num_points,stream_);
      cuda_check(cudaStreamSynchronize(stream_));
      double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
      std::ostringstream out;out<<std::fixed<<std::setprecision(7);
      out<<"{\"sample_idx\":"<<h.sample_idx<<",\"camera_count\":6,\"point_count\":"<<h.num_points
         <<",\"num_detections\":"<<boxes.size()<<",\"backend\":\"native C++/CUDA/TensorRT\",\"model_ms\":"<<ms;
      out<<",\"boxes_3d\":[";
      for(size_t i=0;i<boxes.size();i++){
        auto& x=boxes[i];if(i)out<<",";
        out<<"["<<x.position.x<<","<<x.position.y<<","<<x.position.z<<","<<x.size.w<<","<<x.size.l<<","<<x.size.h
           <<","<<x.z_rotation<<","<<x.velocity.vx<<","<<x.velocity.vy<<"]";
      }
      out<<"],\"scores\":[";
      for(size_t i=0;i<boxes.size();i++){if(i)out<<",";out<<boxes[i].score;}
      out<<"],\"labels\":[";
      for(size_t i=0;i<boxes.size();i++){if(i)out<<",";out<<boxes[i].id;}
      out<<"]}";
      std_msgs::msg::String result;result.data=out.str();pub_->publish(result);
      RCLCPP_INFO(get_logger(),"sample=%u detections=%zu model_ms=%.3f",h.sample_idx,boxes.size(),ms);
    }catch(const std::exception& e){
      RCLCPP_ERROR(get_logger(),"native inference error: %s",e.what());
      std_msgs::msg::String result;result.data=std::string("{\"error\":\"")+e.what()+"\"}";pub_->publish(result);
    }
  }
  std::shared_ptr<bevfusion::Core> core_;
  cudaStream_t stream_{};
  void* image_device_{};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
  rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr sub_;
};
int main(int argc,char**argv){
  rclcpp::init(argc,argv);
  try{rclcpp::spin(std::make_shared<NativeNode>());}
  catch(const std::exception& e){std::fprintf(stderr,"NATIVE_NODE_FAILED %s\\n",e.what());rclcpp::shutdown();return 1;}
  rclcpp::shutdown();return 0;
}
