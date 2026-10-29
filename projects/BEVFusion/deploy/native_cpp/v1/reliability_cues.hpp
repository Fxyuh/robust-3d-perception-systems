#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Same three non-learned cue maps as ReliabilityConvFuser:
// log point density, valid-camera coverage and radial distance.
// Coordinates remain in the unaugmented LiDAR frame (test augmentation is identity).
inline std::vector<float> reliability_cues(const float* points, size_t count,
                                          const float* lidar2image, const float* image_aug,
                                          uint32_t available_mask) {
  constexpr int H=180,W=180,S=H*W;
  std::vector<float> cues(3*S,0.0f);
  for(size_t k=0;k<count;k++){
    const float x=points[k*5],y=points[k*5+1];
    int ix=static_cast<int>(std::floor((x+54.0f)*H/108.0f));
    int iy=static_cast<int>(std::floor((y+54.0f)*W/108.0f));
    if(ix>=0&&ix<H&&iy>=0&&iy<W)cues[ix*W+iy]+=1.0f;
  }
  for(int ix=0;ix<H;ix++)for(int iy=0;iy<W;iy++){
    int i=ix*W+iy;
    cues[i]=std::min(std::log1p(cues[i])/3.0f,1.0f);
    float x=-54.0f+(ix+0.5f)*108.0f/H;
    float y=-54.0f+(iy+0.5f)*108.0f/W;
    int visible=0,available=0;
    for(int camera=0;camera<6;camera++){
      const float* m=lidar2image+camera*16;
      const float* a=image_aug+camera*16;
      float u=m[0]*x+m[1]*y+m[3],v=m[4]*x+m[5]*y+m[7],d=m[8]*x+m[9]*y+m[11];
      float div=std::max(d,1e-5f);
      u/=div;v/=div;
      float aug_u=a[0]*u+a[1]*v+a[2]*d+a[3];
      float aug_v=a[4]*u+a[5]*v+a[6]*d+a[7];
      if(d>1e-5f&&aug_u>=0&&aug_u<704&&aug_v>=0&&aug_v<256){
        visible++;
        if(available_mask&(1u<<camera))available++;
      }
    }
    cues[S+i]=static_cast<float>(available)/std::max(visible,1);
    cues[2*S+i]=std::min(std::sqrt(x*x+y*y)/76.4f,1.0f);
  }
  return cues;
}
