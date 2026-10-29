"use strict";
const recordButton=document.getElementById("record-start"),recordStatus=document.getElementById("record-status");
const videoTypes=["video/mp4;codecs=avc1.640028","video/mp4","video/webm;codecs=vp9","video/webm"];
const videoMime=videoTypes.find(type=>MediaRecorder.isTypeSupported(type));
recordStatus.textContent="编码："+(videoMime||"不支持");
recordButton.onclick=async()=>{
 recordButton.disabled=true;
 try{
  if(!manifest||!current)throw Error("回放数据尚未加载完成");
  stop();threshold=.3;
  const canvas=document.createElement("canvas");canvas.width=1280;canvas.height=720;
  const g=canvas.getContext("2d");document.getElementById("record-preview").append(canvas);
  canvas.style.width="100%";canvas.style.maxWidth="960px";
  const stream=canvas.captureStream(10),chunks=[];
  const recorder=new MediaRecorder(stream,{mimeType:videoMime,videoBitsPerSecond:2400000});
  const stopped=new Promise((resolve,reject)=>{recorder.onstop=resolve;recorder.onerror=reject;});
  recorder.ondataavailable=event=>{if(event.data.size)chunks.push(event.data);};
  function text(value,x,y,size=16,color="#e7eff8"){g.fillStyle=color;g.font=size+"px system-ui";g.fillText(value,x,y);}
  function contain(source,x,y,w,h){const scale=Math.min(w/source.width,h/source.height),W=source.width*scale,H=source.height*scale;g.drawImage(source,x+(w-W)/2,y+(h-H)/2,W,H);}
  function paint(number,total){
   g.fillStyle="#0b1018";g.fillRect(0,0,1280,720);
   text("Robust BEVFusion v1",26,42,29);text("实际 C++ / CUDA / TensorRT 部署结果",26,70,14,"#68d8bd");
   text(manifest.conditions[condition].label,26,102,20);
   const stats=[["当前分支",current.route==="lidar_fallback"?"LiDAR-only 回退":"可靠性门控融合"],
                ["本帧模型耗时",current.model_ms.toFixed(1)+" ms"],
                ["可用相机 / 点数",current.cameras.filter(c=>c.available).length+"/6 · "+current.point_count.toLocaleString()],
                ["81 帧 mAP / NDS",manifest.conditions[condition].metrics.mAP.toFixed(4)+" / "+manifest.conditions[condition].metrics.NDS.toFixed(4)]];
   stats.forEach((item,i)=>{const x=26+i*310;g.fillStyle="#152130";g.fillRect(x,121,295,66);text(item[0],x+12,141,12,"#9bafc4");text(item[1],x+12,172,20,i===0?(current.route==="lidar_fallback"?"#ffd18b":"#61e0bd"):"#e7eff8");});
   text("点云与三维检测框",26,215,17);g.fillStyle="#0e1723";g.fillRect(24,226,650,389);contain(scene,24,226,650,389);
   text("六路相机与投影框",704,215,17);
   current.cameras.forEach((camera,i)=>{const x=704+(i%2)*278,y=230+Math.floor(i/2)*125;
     text(camera.name,x,y,11,"#a7b8ca");text(camera.available?"正常":"输入缺失",x+207,y,11,camera.available?"#61e0bd":"#ffd18b");
     g.drawImage(cameraViews[i].canvas,x,y+8,266,97);});
   text("实际部署精选帧 · sample "+current.sample_idx,26,647,16);
   text("展示抽样点云 / 置信度 ≥ 0.30；完整评测每种条件均为 81 帧",26,674,13,"#9bafc4");
   text("门控融合 + LiDAR 回退",980,647,15,"#68d8bd");
   g.fillStyle="#243449";g.fillRect(26,695,1228,5);g.fillStyle="#61e0bd";g.fillRect(26,695,1228*(number+1)/total,5);
  }
  const selected=Array.from({length:12},(_,i)=>i*3),total=selected.length*3;
  let number=0;
  for(let c=0;c<3;c++){
   condition=c;
   for(const frame of selected){
    index=frame;await loadFrame();paint(number,total);
    if(number===0)recorder.start(1000);
    const png=await new Promise(resolve=>canvas.toBlob(resolve,"image/png"));
    const response=await fetch("/capture-frame?index="+number,{method:"POST",headers:{"Content-Type":"image/png"},body:png});
    if(!response.ok)throw Error("保存视频帧失败");
    recordStatus.textContent="正在录制 "+(number+1)+" / "+total+" · "+videoMime;
    await new Promise(resolve=>setTimeout(resolve,500));number++;
   }
  }
  recorder.stop();await stopped;stream.getTracks().forEach(track=>track.stop());
  const blob=new Blob(chunks,{type:videoMime}),format=videoMime.startsWith("video/mp4")?"mp4":"webm";
  const response=await fetch("/capture-video?format="+format,{method:"POST",headers:{"Content-Type":videoMime},body:blob});
  if(!response.ok)throw Error("保存演示视频失败");
  const result=await response.json();recordStatus.textContent="录制完成："+result.name+" · "+result.bytes+" bytes";
  const video=document.createElement("video");video.controls=true;video.src=result.url;video.style.width="100%";video.style.maxWidth="960px";
  document.getElementById("record-preview").append(video);
 }catch(error){recordStatus.textContent="录制失败："+error.message;}
};
