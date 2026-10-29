#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "bevfusion/bevfusion.hpp"
static std::vector<char> read(const std::string& p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error(p);auto n=f.tellg();std::vector<char> v(n);f.seekg(0);f.read(v.data(),n);return v;}
int main(int argc,char**argv){
 if(argc!=3){std::fprintf(stderr,"usage: native_batch model_dir inputs_root\n");return 2;}
 std::string m=argv[1],d=argv[2];
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
 g.xbound={-54.0,54.0,0.3};g.ybound={-54.0,54.0,0.3};g.zbound={-10.0,10.0,20.0};g.dbound={1.0,60.0,0.5};
 g.image_width=704;g.image_height=256;g.feat_width=88;g.feat_height=32;g.num_camera=6;g.geometry_dim={360,360,80};
 auto& n=p.normalize;
 n.image_width=1600;n.image_height=900;n.output_width=704;n.output_height=256;n.num_camera=6;n.resize_lim=0.48f;
 float mean[3]={0.485,0.456,0.406},stdv[3]={0.229,0.224,0.225};
 n.method=bevfusion::camera::NormMethod::mean_std(mean,stdv,1/255.0f,0.0f);
 auto core=bevfusion::create_core(p);
 if(!core){std::fprintf(stderr,"CORE_CREATE_FAILED\n");return 3;}
 cudaStream_t stream;cudaStreamCreate(&stream);
 void* image_device=nullptr;
 cudaMalloc(&image_device,6ul*3*256*704*sizeof(uint16_t));
 core->set_timer(false);
 for(int frame=0;frame<81;frame++){
  char suffix[8];std::snprintf(suffix,sizeof(suffix),"%03d",frame);
  std::string dir=d+"/"+suffix;
  auto a=read(dir+"/cam2lidar.f32"),b=read(dir+"/cam2img.f32"),c=read(dir+"/lidar2img.f32"),e=read(dir+"/img_aug_matrix.f32");
  auto image=read(dir+"/img.f16"),points=read(dir+"/points.f16");
  if(image.size()!=6ul*3*256*704*sizeof(uint16_t)||points.empty()||points.size()%10){std::fprintf(stderr,"BAD_FRAME %d\n",frame);return 4;}
  auto t0=std::chrono::steady_clock::now();
  core->update(reinterpret_cast<float*>(a.data()),reinterpret_cast<float*>(b.data()),reinterpret_cast<float*>(c.data()),reinterpret_cast<float*>(e.data()),stream);
  cudaMemcpyAsync(image_device,image.data(),image.size(),cudaMemcpyHostToDevice,stream);
  auto boxes=core->forward_no_normalize(reinterpret_cast<const nvtype::half*>(image_device),reinterpret_cast<const nvtype::half*>(points.data()),points.size()/10,stream);
  cudaStreamSynchronize(stream);
  auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
  std::printf("FRAME %03d %zu %.3f\n",frame,boxes.size(),ms);
  for(auto& x:boxes)std::printf("BOX %d %.8f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",x.id,x.score,x.position.x,x.position.y,x.position.z,x.size.w,x.size.l,x.size.h,x.velocity.vx,x.velocity.vy,x.z_rotation);
  std::printf("END_FRAME %03d\n",frame);
  std::fflush(stdout);
 }
 cudaFree(image_device);cudaStreamDestroy(stream);
 std::printf("NATIVE_FRAMES 81\n");
 return 0;
}
