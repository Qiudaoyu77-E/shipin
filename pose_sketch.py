#!/usr/bin/env python3
"""将照片中的人物提取为绘画练习用骨架姿势图（支持单张/批量/网页）。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8-pose COCO 17 keypoints连接关系
COCO_EDGES: Sequence[tuple[int, int]] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把人物照片输出成骨架姿势图，用于绘画临摹。"
    )
    parser.add_argument("--input", required=True, help="输入图片路径，或目录路径（批量模式）")
    parser.add_argument("--output", required=True, help="输出图片路径，或目录路径（批量模式）")
    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="YOLO pose 模型名称或本地模型路径（默认: yolov8n-pose.pt）",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="人体检测阈值（默认: 0.25）")
    parser.add_argument(
        "--kpt-conf", type=float, default=0.4, help="关键点置信度阈值（默认: 0.4）"
    )
    parser.add_argument(
        "--line-thickness", type=int, default=3, help="骨架线条粗细（默认: 3）"
    )
    parser.add_argument(
        "--point-radius", type=int, default=4, help="关键点半径（默认: 4）"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="启用批量处理：--input/--output 都应为目录",
    )
    return parser.parse_args()


def iter_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            yield p


def draw_pose_canvas(
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    kpt_conf_thres: float,
    line_thickness: int,
    point_radius: int,
) -> np.ndarray:
    h, w = image.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    for person_xy, person_conf in zip(keypoints_xy, keypoints_conf):
        for i, j in COCO_EDGES:
            if person_conf[i] >= kpt_conf_thres and person_conf[j] >= kpt_conf_thres:
                p1 = tuple(person_xy[i].astype(int))
                p2 = tuple(person_xy[j].astype(int))
                cv2.line(canvas, p1, p2, (0, 0, 0), line_thickness, cv2.LINE_AA)

        for idx, pt in enumerate(person_xy):
            if person_conf[idx] >= kpt_conf_thres:
                cv2.circle(
                    canvas,
                    tuple(pt.astype(int)),
                    point_radius,
                    (0, 0, 255),
                    -1,
                    cv2.LINE_AA,
                )

    return canvas


def extract_pose_image(
    model: YOLO,
    image: np.ndarray,
    conf: float = 0.25,
    kpt_conf: float = 0.4,
    line_thickness: int = 3,
    point_radius: int = 4,
) -> np.ndarray:
    """输入BGR图像，返回骨架图(BGR)。"""
    result = model.predict(source=image, conf=conf, verbose=False)[0]
    if result.keypoints is None or result.keypoints.xy is None:
        return np.full_like(image, 255)

    keypoints_xy = result.keypoints.xy.cpu().numpy()
    if result.keypoints.conf is None:
        keypoints_conf = np.ones(keypoints_xy.shape[:2], dtype=np.float32)
    else:
        keypoints_conf = result.keypoints.conf.cpu().numpy()

    return draw_pose_canvas(
        image=image,
        keypoints_xy=keypoints_xy,
        keypoints_conf=keypoints_conf,
        kpt_conf_thres=kpt_conf,
        line_thickness=line_thickness,
        point_radius=point_radius,
    )


def process_one(
    model: YOLO,
    input_path: Path,
    output_path: Path,
    conf: float,
    kpt_conf: float,
    line_thickness: int,
    point_radius: int,
) -> None:
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"无法读取图片: {input_path}")

    canvas = extract_pose_image(
        model=model,
        image=image,
        conf=conf,
        kpt_conf=kpt_conf,
        line_thickness=line_thickness,
        point_radius=point_radius,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), canvas)
    if not ok:
        raise RuntimeError(f"写入失败: {output_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    model = YOLO(args.model)

    if args.batch:
        if not input_path.is_dir():
            raise NotADirectoryError("批量模式下 --input 必须是目录")
        output_path.mkdir(parents=True, exist_ok=True)

        images = list(iter_images(input_path))
        if not images:
            raise FileNotFoundError(f"目录中未找到图片: {input_path}")

        for img in images:
            out = output_path / f"{img.stem}_pose.png"
            process_one(
                model=model,
                input_path=img,
                output_path=out,
                conf=args.conf,
                kpt_conf=args.kpt_conf,
                line_thickness=args.line_thickness,
                point_radius=args.point_radius,
            )
            print(f"[OK] {img.name} -> {out}")
    else:
        if not input_path.is_file():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        if output_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("单图模式下 --output 必须是图片文件路径（例如 output.png）")

        process_one(
            model=model,
            input_path=input_path,
            output_path=output_path,
            conf=args.conf,
            kpt_conf=args.kpt_conf,
            line_thickness=args.line_thickness,
            point_radius=args.point_radius,
        )
        print(f"[OK] {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
