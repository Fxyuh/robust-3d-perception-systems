#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>
#include "v1_core.hpp"

struct V1PacketHeader{
  char magic[4];uint32_t version,sample_idx,num_points,image_bytes,point_bytes,matrix_bytes,available_mask;
};
static_assert(sizeof(V1PacketHeader)==32);
class V1Node:public rclcpp::Node{
 public:
  V1Node():Node("bevfusion_v1_native"){
    auto model=declare_parameter<std::string>("model_dir","");
    if(model.empty())throw std::runtime_error("Pass model_dir for the frozen v1 exports");
    core_=std::make_unique<V1Core>(model);
    pub_=create_publisher<std_msgs::msg::String>("/bevfusion/v1_detections_json",1);
    sub_=create_subscription<std_msgs::msg::UInt8MultiArray>("/bevfusion/v1_input",1,
      [this](std_msgs::msg::UInt8MultiArray::SharedPtr msg){on_packet(*msg);});
    RCLCPP_INFO(get_logger(),"Frozen v1 native ready: gated fusion and LiDAR fallback resident");
  }
 private:
  void on_packet(const std_msgs::msg::UInt8MultiArray& msg){
    try{
      if(msg.data.size()<32)throw std::runtime_error("packet too short");
      V1PacketHeader h;std::memcpy(&h,msg.data.data(),32);
      if(std::memcmp(h.magic,"BFV1",4)||h.version!=2||h.available_mask>63||
         !h.num_points||h.num_points>300000||h.image_bytes!=6ul*3*256*704*2||
         h.point_bytes!=h.num_points*5*sizeof(float)||h.matrix_bytes!=1536||
         msg.data.size()!=32ul+h.image_bytes+h.point_bytes+h.matrix_bytes)
        throw std::runtime_error("invalid v1 packet contract");
      const uint8_t* image=msg.data.data()+32;
      const uint8_t* points=image+h.image_bytes;
      const uint8_t* matrices=points+h.point_bytes;
      std::vector<float> cloud(h.num_points*5);
      std::array<float,384> calib;
      std::memcpy(cloud.data(),points,h.point_bytes);
      std::memcpy(calib.data(),matrices,h.matrix_bytes);
      auto started=std::chrono::steady_clock::now();
      auto boxes=core_->forward(reinterpret_cast<const uint16_t*>(image),cloud.data(),h.num_points,
                               calib.data(),calib.data()+96,calib.data()+192,calib.data()+288,h.available_mask);
      double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
      std::ostringstream out;out<<std::fixed<<std::setprecision(8);
      out<<"{\"sample_idx\":"<<h.sample_idx<<",\"camera_count\":6,\"available_mask\":"<<h.available_mask
         <<",\"point_count\":"<<h.num_points<<",\"num_detections\":"<<boxes.size()
         <<",\"route\":\""<<(h.available_mask?"gated_fusion":"lidar_fallback")
         <<"\",\"backend\":\"frozen v1 native C++/CUDA/TensorRT\",\"model_ms\":"<<ms<<",\"boxes_3d\":[";
      for(size_t i=0;i<boxes.size();i++){auto& b=boxes[i];if(i)out<<",";
        out<<"["<<b.position.x<<","<<b.position.y<<","<<b.position.z<<","<<b.size.w<<","<<b.size.l<<","<<b.size.h
           <<","<<b.z_rotation<<","<<b.velocity.vx<<","<<b.velocity.vy<<"]";}
      out<<"],\"scores\":[";
      for(size_t i=0;i<boxes.size();i++){if(i)out<<",";out<<boxes[i].score;}
      out<<"],\"labels\":[";
      for(size_t i=0;i<boxes.size();i++){if(i)out<<",";out<<boxes[i].id;}
      out<<"]}";std_msgs::msg::String result;result.data=out.str();pub_->publish(result);
      RCLCPP_INFO(get_logger(),"sample=%u route=%s model_ms=%.3f",h.sample_idx,h.available_mask?"gate":"lidar",ms);
    }catch(const std::exception& e){
      RCLCPP_ERROR(get_logger(),"%s",e.what());
      std_msgs::msg::String result;result.data="{\"error\":\"native v1 inference failed\"}";pub_->publish(result);
    }
  }
  std::unique_ptr<V1Core> core_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
  rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr sub_;
};
int main(int argc,char**argv){
  rclcpp::init(argc,argv);
  try{rclcpp::spin(std::make_shared<V1Node>());}
  catch(const std::exception& e){std::fprintf(stderr,"V1_NODE_FAILED %s\n",e.what());rclcpp::shutdown();return 1;}
  rclcpp::shutdown();return 0;
}
