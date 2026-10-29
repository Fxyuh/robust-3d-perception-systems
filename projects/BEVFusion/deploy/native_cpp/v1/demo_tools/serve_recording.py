"""Local-only recorder for the project's own canvas replay; no screen/microphone capture."""
import argparse,json
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--viewer-dir",type=Path,required=True)
p.add_argument("--output-dir",type=Path,required=True)
p.add_argument("--offline-demo",type=Path,required=True)
p.add_argument("--port",type=int,default=8770)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);(a.output_dir/"frames").mkdir()
source=Path(__file__).resolve().parent
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(a.viewer_dir),**kwargs)
    def content(self,data,mime):
        self.send_response(200);self.send_header("Content-Type",mime);self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=="/record.html":
            html=(a.viewer_dir/"index.html").read_text()
            html=html.replace("<main>",'<main><section style="padding:14px;background:#142331"><button id="record-start">录制仓库演示</button> <span id="record-status"></span><div id="record-preview"></div></section>',1)
            html=html.replace("</body>",'<script src="/record.js"></script></body>')
            return self.content(html.encode(),"text/html; charset=utf-8")
        if path=="/record.js":return self.content((source/"record.js").read_bytes(),"application/javascript")
        if path=="/offline.html":return self.content(a.offline_demo.read_bytes(),"text/html; charset=utf-8")
        if path.startswith("/exported/"):
            name=path.rsplit("/",1)[-1]
            if name not in ("bevfusion_v1_demo.mp4","bevfusion_v1_demo.webm"):return self.send_error(404)
            file=a.output_dir/name
            if not file.exists():return self.send_error(404)
            return self.content(file.read_bytes(),"video/mp4" if file.suffix==".mp4" else "video/webm")
        return super().do_GET()
    def do_POST(self):
        origin=self.headers.get("Origin","")
        if origin not in (f"http://localhost:{a.port}",f"http://127.0.0.1:{a.port}"):return self.send_error(403)
        length=int(self.headers.get("Content-Length","0"))
        if not 0<length<50*1024*1024:return self.send_error(413)
        path=urlsplit(self.path);query=parse_qs(path.query)
        if path.path=="/capture-frame":
            try:index=int(query["index"][0])
            except (KeyError,ValueError):return self.send_error(400)
            if not 0<=index<100:return self.send_error(400)
            file=a.output_dir/"frames"/f"{index:03d}.png"
        elif path.path=="/capture-video":
            format=query.get("format",[""])[0]
            if format not in ("mp4","webm"):return self.send_error(400)
            file=a.output_dir/("bevfusion_v1_demo."+format)
        else:return self.send_error(404)
        if file.exists():return self.send_error(409)
        file.write_bytes(self.rfile.read(length))
        return self.content(json.dumps({"name":file.name,"bytes":file.stat().st_size,"url":"/exported/"+file.name}).encode(),"application/json")
ThreadingHTTPServer(("127.0.0.1",a.port),Handler).serve_forever()
