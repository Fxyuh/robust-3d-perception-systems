"use strict";
const $=id=>document.getElementById(id);
const classes=["car","truck","construction_vehicle","bus","trailer","barrier","motorcycle","bicycle","pedestrian","traffic_cone"];
const colors=["#4edbc6","#67a4ff","#e49eff","#ffab6b","#d7abff","#abc0d3","#ff83b6","#aa9cff","#ffdc78","#ff9c62"];
let manifest,condition=0,index=0,current=null,threshold=.3,azimuth=-.85,elevation=1.05,zoom=1,playing=null,requestId=0;
let cameraViews=[];
const scene=$("scene"),ctx=scene.getContext("2d");
function corners(b){const [label,score,x,y,z,dx,dy,dz,yaw]=b,c=Math.cos(yaw),s=Math.sin(yaw);return [[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]].map(([a,d,h])=>[x+c*a*dx/2-s*d*dy/2,y+s*a*dx/2+c*d*dy/2,z+h*dz]);}
const edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
function project3(p,w,h){let c=Math.cos(azimuth),s=Math.sin(azimuth),x=p[0]*c-p[1]*s,y=p[0]*s+p[1]*c,scale=Math.min(w,h)/95*zoom;return [w*.5+x*scale,h*.53-(y*Math.sin(elevation)+p[2]*Math.cos(elevation))*scale];}
function line3(a,b,color,width=1){const w=scene.clientWidth,h=scene.clientHeight,A=project3(a,w,h),B=project3(b,w,h);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...A);ctx.lineTo(...B);ctx.stroke();}
function renderScene(){
 const w=scene.clientWidth,h=scene.clientHeight,dpr=Math.min(window.devicePixelRatio||1,2);
 if(scene.width!==Math.round(w*dpr)||scene.height!==Math.round(h*dpr)){scene.width=Math.round(w*dpr);scene.height=Math.round(h*dpr);}
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle="#0e1723";ctx.fillRect(0,0,w,h);
 for(let t=-50;t<=50;t+=10){line3([t,-54,-2],[t,54,-2],"#203044");line3([-54,t,-2],[54,t,-2],"#203044");}
 if(!current)return;
 const pts=current.points;
 for(let i=0;i<pts.length;i+=3){const p=project3([pts[i],pts[i+1],pts[i+2]],w,h);if(p[0]<0||p[0]>w||p[1]<0||p[1]>h)continue;
 const z=Math.max(0,Math.min(1,(pts[i+2]+2.5)/5));ctx.fillStyle=z<.25?"#4a7795":z<.6?"#66b9ad":"#e5cf85";ctx.globalAlpha=.64;ctx.fillRect(p[0],p[1],1.6,1.6);}
 ctx.globalAlpha=1;
 const boxes=current.boxes.filter(b=>b[1]>=threshold);
 for(const b of boxes){const c=corners(b),color=colors[b[0]];for(const [a,d] of edges)line3(c[a],c[d],color,1.5);
 const pos=project3([b[2],b[3],b[4]+b[7]+.4],w,h);ctx.fillStyle=color;ctx.font="10px system-ui";if(b[1]>=.5)ctx.fillText(classes[b[0]]+" "+b[1].toFixed(2),pos[0]+2,pos[1]-3);}
 line3([0,0,-1.6],[4,0,-1.6],"#edf4fc",2.5);line3([0,0,-1.6],[0,4,-1.6],"#7e9cb9",2);
 const ego=project3([0,0,-1.6],w,h);ctx.fillStyle="#edf4fc";ctx.beginPath();ctx.arc(...ego,4,0,Math.PI*2);ctx.fill();
 $("visible-count").textContent="展示 "+Math.floor(pts.length/3).toLocaleString()+" 点 · "+boxes.length+" 个框";
}
function projectCamera(p,camera){
 const m=camera.lidar2img,a=camera.augmentation;
 const X=m[0]*p[0]+m[1]*p[1]+m[2]*p[2]+m[3],Y=m[4]*p[0]+m[5]*p[1]+m[6]*p[2]+m[7],d=m[8]*p[0]+m[9]*p[1]+m[10]*p[2]+m[11];
 if(d<=.1)return null;const u=X/d,v=Y/d;
 return [a[0]*u+a[1]*v+a[2]*d+a[3],a[4]*u+a[5]*v+a[6]*d+a[7]];
}
function renderCameras(){
 if(!current)return;
 current.cameras.forEach((camera,i)=>{
 const {canvas,image,label}=cameraViews[i],c=canvas.getContext("2d");
 c.clearRect(0,0,704,256);
 label.textContent=camera.available?"正常":"输入缺失";label.className=camera.available?"ok":"off";
 if(!camera.available){c.fillStyle="#202b38";c.fillRect(0,0,704,256);c.strokeStyle="#3d4c60";c.lineWidth=2;c.beginPath();c.moveTo(0,0);c.lineTo(704,256);c.moveTo(704,0);c.lineTo(0,256);c.stroke();c.fillStyle="#c9d1dc";c.font="22px system-ui";c.textAlign="center";c.fillText("相机信号缺失",352,139);c.textAlign="left";return;}
 if(image.complete&&image.naturalWidth)c.drawImage(image,0,0,704,256);
 for(const b of current.boxes){if(b[1]<threshold)continue;const p=corners(b).map(x=>projectCamera(x,camera));if(p.some(x=>!x))continue;
 if(p.every(x=>x[0]<0)||p.every(x=>x[0]>704)||p.every(x=>x[1]<0)||p.every(x=>x[1]>256))continue;
 c.strokeStyle=colors[b[0]];c.lineWidth=1.4;c.beginPath();for(const [a,d] of edges){c.moveTo(...p[a]);c.lineTo(...p[d]);}c.stroke();}
 });
}
async function loadFrame(){
 const id=++requestId;
 try{
 const spec=manifest.conditions[condition],frame=spec.frames[index];
 const response=await fetch(frame);if(!response.ok)throw Error("回放帧读取失败");
 const data=await response.json();if(id!==requestId)return;
 const images=await Promise.all(data.cameras.map(camera=>new Promise(resolve=>{const im=new Image();if(!camera.image){resolve(im);return;}im.onload=()=>resolve(im);im.onerror=()=>resolve(im);im.src=camera.image;})));
 if(id!==requestId)return;current=data;images.forEach((im,i)=>cameraViews[i].image=im);
 $("route").textContent=data.route==="lidar_fallback"?"LiDAR-only 回退":"可靠性门控融合";$("route").classList.toggle("fallback",data.route==="lidar_fallback");
 $("latency").textContent=data.model_ms.toFixed(1)+" ms";
 $("sensors").textContent=data.cameras.filter(c=>c.available).length+" / 6 · "+data.point_count.toLocaleString()+" 点";
 $("metrics").textContent=spec.metrics.mAP.toFixed(4)+" / "+spec.metrics.NDS.toFixed(4);
 $("frame").value=index;$("frame-label").textContent="帧 "+(index+1)+" / "+spec.frames.length+" · sample "+data.sample_idx;
 $("status").textContent="已加载实际检测结果";$("status").classList.remove("error");renderScene();renderCameras();
 }catch(error){$("status").textContent=error.message;$("status").classList.add("error");stop();}
}
function stop(){if(playing){clearInterval(playing);playing=null;}$("play").textContent="▶ 播放";}
$("play").onclick=()=>{if(playing){stop();return;}$("play").textContent="Ⅱ 暂停";playing=setInterval(()=>{index=(index+1)%manifest.conditions[condition].frames.length;loadFrame();},500);};
$("frame").oninput=event=>{stop();index=+event.target.value;loadFrame();};
$("threshold").oninput=event=>{threshold=+event.target.value;$("threshold-value").textContent=threshold.toFixed(2);renderScene();renderCameras();};
$("top").onclick=()=>{elevation=Math.PI/2;azimuth=-Math.PI/2;renderScene();};
$("reset").onclick=()=>{azimuth=-.85;elevation=1.05;zoom=1;renderScene();};
let drag=null;
scene.onpointerdown=event=>{drag={x:event.clientX,y:event.clientY};scene.setPointerCapture(event.pointerId);};
scene.onpointermove=event=>{if(!drag)return;azimuth+=(event.clientX-drag.x)*.007;elevation=Math.max(.1,Math.min(1.56,elevation+(event.clientY-drag.y)*.006));drag={x:event.clientX,y:event.clientY};renderScene();};
scene.onpointerup=scene.onpointercancel=()=>drag=null;
scene.addEventListener("wheel",event=>{event.preventDefault();zoom=Math.max(.4,Math.min(4,zoom*Math.exp(-event.deltaY*.001)));renderScene();},{passive:false});
window.addEventListener("resize",renderScene);
async function initialize(){
 try{
 const response=await fetch("manifest.json");if(!response.ok)throw Error("没有找到本地回放数据");manifest=await response.json();
 const names=manifest.camera_names;
 cameraViews=names.map(name=>{const card=document.createElement("div");card.className="camera";const title=document.createElement("div");title.className="label";const text=document.createElement("span");text.textContent=name;const label=document.createElement("span");title.append(text,label);const canvas=document.createElement("canvas");canvas.width=704;canvas.height=256;canvas.setAttribute("aria-label",name+" 相机与检测框");card.append(title,canvas);$("cameras").append(card);return{canvas,label,image:new Image()};});
 manifest.conditions.forEach((spec,i)=>{const button=document.createElement("button");button.textContent=spec.label;button.classList.toggle("active",i===0);button.setAttribute("aria-pressed",i===0?"true":"false");button.onclick=()=>{stop();condition=i;index=Math.min(index,spec.frames.length-1);[...$("conditions").children].forEach((b,j)=>{b.classList.toggle("active",j===i);b.setAttribute("aria-pressed",j===i?"true":"false");});$("frame").max=spec.frames.length-1;loadFrame();};$("conditions").append(button);});
 $("frame").max=manifest.conditions[0].frames.length-1;await loadFrame();
 }catch(error){$("status").textContent=error.message;$("status").classList.add("error");}
}
initialize();
