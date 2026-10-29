#include <fstream>
#include <string>
#include <vector>
#include "reliability_cues.hpp"
template<class T>std::vector<T> read(const std::string& name){
 std::ifstream f(name,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error(name);
 auto size=f.tellg();std::vector<T> values(static_cast<size_t>(size)/sizeof(T));
 f.seekg(0);f.read(reinterpret_cast<char*>(values.data()),size);return values;
}
int main(int argc,char**argv){
 if(argc!=4)return 2;
 std::string d=argv[1];
 auto points=read<float>(d+"/points.f32"),l2i=read<float>(d+"/lidar2img.f32"),aug=read<float>(d+"/img_aug_matrix.f32");
 auto cues=reliability_cues(points.data(),points.size()/5,l2i.data(),aug.data(),std::stoul(argv[2]));
 std::ofstream out(argv[3],std::ios::binary);out.write(reinterpret_cast<char*>(cues.data()),cues.size()*sizeof(float));
}
