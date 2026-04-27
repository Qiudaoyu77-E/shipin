#!/usr/bin/env python3
"""本地网页：上传单张或多张图片，输出骨架姿势图。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

from pose_sketch import extract_pose_image

MODEL_NAME = "yolov8n-pose.pt"
_model = YOLO(MODEL_NAME)


def _to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def run_single(
    image: np.ndarray,
    conf: float,
    kpt_conf: float,
    line_thickness: int,
    point_radius: int,
) -> np.ndarray:
    if image is None:
        raise gr.Error("请先上传图片")

    bgr = _to_bgr(image)
    out_bgr = extract_pose_image(
        model=_model,
        image=bgr,
        conf=conf,
        kpt_conf=kpt_conf,
        line_thickness=line_thickness,
        point_radius=point_radius,
    )
    return _to_rgb(out_bgr)


def run_batch(
    files: List[str],
    conf: float,
    kpt_conf: float,
    line_thickness: int,
    point_radius: int,
):
    if not files:
        raise gr.Error("请先上传至少一张图片")

    out_paths = []
    out_dir = Path(tempfile.mkdtemp(prefix="pose_batch_"))

    for file in files:
        src = Path(file)
        bgr = cv2.imread(str(src))
        if bgr is None:
            continue

        out_bgr = extract_pose_image(
            model=_model,
            image=bgr,
            conf=conf,
            kpt_conf=kpt_conf,
            line_thickness=line_thickness,
            point_radius=point_radius,
        )
        out_file = out_dir / f"{src.stem}_pose.png"
        cv2.imwrite(str(out_file), out_bgr)
        out_paths.append(str(out_file))

    if not out_paths:
        raise gr.Error("上传的文件里没有可读图片")

    return out_paths


with gr.Blocks(title="照片转绘画骨架/姿势图") as demo:
    gr.Markdown("# 照片转绘画骨架/姿势图\n支持单张和批量处理（本地网页运行）")

    with gr.Row():
        conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="人体检测阈值")
        kpt_conf = gr.Slider(0.05, 0.95, value=0.4, step=0.05, label="关键点阈值")
        line_thickness = gr.Slider(1, 10, value=3, step=1, label="线条粗细")
        point_radius = gr.Slider(1, 12, value=4, step=1, label="关键点大小")

    with gr.Tab("单张图片"):
        in_img = gr.Image(type="numpy", label="上传照片")
        out_img = gr.Image(type="numpy", label="骨架结果")
        btn_single = gr.Button("生成骨架图")
        btn_single.click(
            fn=run_single,
            inputs=[in_img, conf, kpt_conf, line_thickness, point_radius],
            outputs=out_img,
        )

    with gr.Tab("批量图片"):
        in_files = gr.File(file_count="multiple", file_types=["image"], label="上传多张图片")
        out_files = gr.Files(label="下载结果")
        btn_batch = gr.Button("批量生成")
        btn_batch.click(
            fn=run_batch,
            inputs=[in_files, conf, kpt_conf, line_thickness, point_radius],
            outputs=out_files,
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
