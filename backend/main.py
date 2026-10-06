"""FastAPI 服务：曝光包围去重影合成台后端。"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline import PipelineError, run_pipeline

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = Path(os.environ.get("FRONTEND_DIR", BASE_DIR.parent / "frontend"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", BASE_DIR / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="曝光包围去重影合成台")

ARTIFACTS = {
    "result": "result.png",
    "mask": "mask.png",
    "reference": "reference.png",
    "meta": "meta.json",
}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/merge")
async def merge(
    files: list[UploadFile] = File(...),
    shutter_times: str = Form(...),
    reference_index: int = Form(...),
):
    """上传 3～7 张 JPEG + 各自快门时间（秒）+ 参照帧索引，执行合成。"""
    try:
        times = json.loads(shutter_times)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(400, "快门时间格式非法：应为 JSON 数组，如 [0.004, 0.016, 0.067]")
    if not isinstance(times, list):
        raise HTTPException(400, "快门时间格式非法：应为 JSON 数组")

    raw = []
    for f in files:
        data = await f.read()
        if not data:
            raise HTTPException(400, f"文件 {f.filename or '?'} 为空")
        raw.append(data)

    try:
        result = run_pipeline(raw, times, reference_index)
    except PipelineError as e:
        raise HTTPException(400, str(e))

    job_id = uuid.uuid4().hex[:12]
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    (job_dir / "result.png").write_bytes(result["result_png"])
    (job_dir / "mask.png").write_bytes(result["mask_png"])
    (job_dir / "reference.png").write_bytes(result["reference_png"])
    (job_dir / "meta.json").write_text(
        json.dumps(result["meta"], ensure_ascii=False, indent=2))
    return {"job_id": job_id, "meta": result["meta"]}


@app.get("/api/jobs/{job_id}/{kind}")
def get_artifact(job_id: str, kind: str):
    """下载合成产物：result（成片 PNG）/ mask（不确定遮罩）/ reference（参照帧）/ meta。"""
    if kind not in ARTIFACTS or not job_id.isalnum():
        raise HTTPException(404, "产物不存在")
    path = RESULTS_DIR / job_id / ARTIFACTS[kind]
    if not path.is_file():
        raise HTTPException(404, "产物不存在")
    return FileResponse(path)


# 静态前端（挂在最后，/api 路由优先匹配）
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
