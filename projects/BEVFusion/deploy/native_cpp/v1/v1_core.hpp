// Native module orchestration follows NVIDIA CUDA-BEVFusion (MIT),
// commit 7c1623f234d5a9794fd862c9456f68f87ae7b900.
// The upstream CUDA algorithms are external; this wrapper adds v1 cue input
// and explicit two-expert routing without modifying the upstream library.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "bevfusion/bevfusion.hpp"
#include "common/tensorrt.hpp"
#include "reliability_cues.hpp"

inline void v1_cuda(cudaError_t code){
  if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}
inline bevfusion::CoreParameter v1_parameters(const std::string& dir){
  bevfusion::CoreParameter p;
  p.camera_model=dir+"/camera.engine";
  p.camera_vtransform=dir+"/camera_vtransform.engine";
  p.transfusion=dir+"/fuser.engine";
  p.transbbox.model=dir+"/head_bbox.engine";
  p.transbbox.confidence_threshold=0.0f;p.transbbox.sorted_bboxes=true;
  p.transbbox.out_size_factor=8;p.transbbox.voxel_size={0.075,0.075};
  p.transbbox.pc_range={-54.0,-54.0};
  p.transbbox.post_center_range_start={-61.2,-61.2,-10.0};
  p.transbbox.post_center_range_end={61.2,61.2,10.0};
  auto& v=p.lidar_scn.voxelization;
  v.min_range={-54,-54,-5};v.max_range={54,54,3};v.voxel_size={0.075,0.075,0.2};
  v.grid_size=v.compute_grid_size(v.max_range,v.min_range,v.voxel_size);
  v.max_points_per_voxel=10;v.max_points=300000;v.max_voxels=160000;v.num_feature=5;
  p.lidar_scn.model=dir+"/lidar_sparse.onnx";
  p.lidar_scn.order=bevfusion::lidar::CoordinateOrder::XYZ;
  p.lidar_scn.precision=bevfusion::lidar::Precision::Float16;
  auto& g=p.geometry;
  g.xbound={-54,54,0.3};g.ybound={-54,54,0.3};g.zbound={-10,10,20};g.dbound={1,60,0.5};
  g.image_width=704;g.image_height=256;g.feat_width=88;g.feat_height=32;g.num_camera=6;g.geometry_dim={360,360,80};
  return p;
}
class V1Core{
 public:
  explicit V1Core(const std::string& dir){
    auto f=v1_parameters(dir+"/fusion"),l=v1_parameters(dir+"/lidar");
    fusion_lidar_=bevfusion::lidar::create_scn(f.lidar_scn);
    fallback_lidar_=bevfusion::lidar::create_scn(l.lidar_scn);
    camera_=bevfusion::camera::create_backbone(f.camera_model);
    if(!camera_)throw std::runtime_error("camera engine initialization failed");
    pool_=bevfusion::camera::create_bevpool(camera_->camera_shape(),360,360);
    downsample_=bevfusion::camera::create_vtransform(f.camera_vtransform);
    depth_=bevfusion::camera::create_depth(704,256,6);
    geometry_=bevfusion::camera::create_geometry(f.geometry);
    gate_=TensorRT::load(f.transfusion);
    fallback_trunk_=TensorRT::load(dir+"/lidar/trunk.engine");
    fusion_head_=bevfusion::head::transbbox::create_transbbox(f.transbbox);
    fallback_head_=bevfusion::head::transbbox::create_transbbox(l.transbbox);
    if(!fusion_lidar_||!fallback_lidar_||!pool_||!downsample_||!depth_||!geometry_||
       !gate_||!fallback_trunk_||!fusion_head_||!fallback_head_)
      throw std::runtime_error("v1 native module initialization failed");
    if(gate_->dtype("reliability_cues")!=TensorRT::DType::FLOAT)
      throw std::runtime_error("v1 gate requires FP32 reliability cue binding");
    v1_cuda(cudaStreamCreate(&stream_));
    v1_cuda(cudaMalloc(&images_,6ul*3*256*704*sizeof(__half)));
    v1_cuda(cudaMalloc(&points_,300000ul*5*sizeof(__half)));
    v1_cuda(cudaMalloc(&cues_,3ul*180*180*sizeof(float)));
    v1_cuda(cudaMalloc(&middle_,512ul*180*180*sizeof(__half)));
  }
  ~V1Core(){
    if(images_)cudaFree(images_);if(points_)cudaFree(points_);
    if(cues_)cudaFree(cues_);if(middle_)cudaFree(middle_);
    if(stream_)cudaStreamDestroy(stream_);
  }
  std::vector<bevfusion::head::transbbox::BoundingBox> forward(
      const uint16_t* image,const float* points,size_t count,
      const float* c2l,const float* intrinsic,const float* l2i,const float* aug,uint32_t mask){
    if(!count||count>300000||mask>63)throw std::runtime_error("Invalid v1 inputs");
    std::vector<__half> converted(count*5);
    for(size_t i=0;i<converted.size();i++)converted[i]=__float2half(points[i]);
    v1_cuda(cudaMemcpyAsync(points_,converted.data(),converted.size()*2,cudaMemcpyHostToDevice,stream_));
    const auto* cloud=reinterpret_cast<const nvtype::half*>(points_);
    if(mask==0){
      auto lidar=fallback_lidar_->forward(cloud,count,stream_);
      if(!fallback_trunk_->forward({{"lidar",lidar},{"middle",middle_}},stream_))
        throw std::runtime_error("fallback trunk inference failed");
      auto boxes=fallback_head_->forward(reinterpret_cast<const nvtype::half*>(middle_),0.0f,stream_,true);
      v1_cuda(cudaStreamSynchronize(stream_));
      return boxes;
    }
    auto cue=reliability_cues(points,count,l2i,aug,mask);
    v1_cuda(cudaMemcpyAsync(cues_,cue.data(),cue.size()*sizeof(float),cudaMemcpyHostToDevice,stream_));
    v1_cuda(cudaMemcpyAsync(images_,image,6ul*3*256*704*sizeof(__half),cudaMemcpyHostToDevice,stream_));
    depth_->update(aug,l2i,stream_);geometry_->update(c2l,intrinsic,aug,stream_);
    auto lidar=fusion_lidar_->forward(cloud,count,stream_);
    auto d=depth_->forward(cloud,count,5,stream_);
    camera_->forward(reinterpret_cast<const nvtype::half*>(images_),d,stream_);
    auto pooled=pool_->forward(camera_->feature(),camera_->depth(),geometry_->indices(),
                              geometry_->intervals(),geometry_->num_intervals(),stream_);
    auto camera=downsample_->forward(pooled,stream_);
    if(!gate_->forward({{"camera",camera},{"lidar",lidar},{"reliability_cues",cues_},{"middle",middle_}},stream_))
      throw std::runtime_error("gated fusion trunk inference failed");
    auto boxes=fusion_head_->forward(reinterpret_cast<const nvtype::half*>(middle_),0.0f,stream_,true);
    v1_cuda(cudaStreamSynchronize(stream_));
    return boxes;
  }
 private:
  cudaStream_t stream_{};
  void *images_=nullptr,*points_=nullptr,*cues_=nullptr,*middle_=nullptr;
  std::shared_ptr<bevfusion::lidar::SCN> fusion_lidar_,fallback_lidar_;
  std::shared_ptr<bevfusion::camera::Backbone> camera_;
  std::shared_ptr<bevfusion::camera::BEVPool> pool_;
  std::shared_ptr<bevfusion::camera::VTransform> downsample_;
  std::shared_ptr<bevfusion::camera::Depth> depth_;
  std::shared_ptr<bevfusion::camera::Geometry> geometry_;
  std::shared_ptr<TensorRT::Engine> gate_,fallback_trunk_;
  std::shared_ptr<bevfusion::head::transbbox::TransBBox> fusion_head_,fallback_head_;
};
