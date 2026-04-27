#!/usr/bin/env python3
"""将照片中的人物提取为绘画练习用骨架姿势图（支持单张/批量/网页）。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from ultralytics import YOLO
try:
    import mediapipe as mp
except Exception:  # noqa: BLE001
    mp = None

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
    parser.add_argument(
        "--no-mp-fallback",
        action="store_true",
        help="关闭 MediaPipe 回退（默认开启，用于半身/遮挡场景）",
    )
    return parser.parse_args()


def mp_pose_to_coco17(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """使用 MediaPipe Pose 估算关键点并映射为 COCO17。"""
    if mp is None:
        return None, None

    mapping = {
        0: 0,   # nose
        5: 11,  # left_shoulder
        6: 12,  # right_shoulder
        7: 13,  # left_elbow
        8: 14,  # right_elbow
        9: 15,  # left_wrist
        10: 16, # right_wrist
        11: 23, # left_hip
        12: 24, # right_hip
        13: 25, # left_knee
        14: 26, # right_knee
        15: 27, # left_ankle
        16: 28, # right_ankle
        1: 2,   # left_eye
        2: 5,   # right_eye
        3: 7,   # left_ear
        4: 8,   # right_ear
    }

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=2, enable_segmentation=False
    ) as pose:
        res = pose.process(rgb)
    if res.pose_landmarks is None:
        return None, None

    h, w = image_bgr.shape[:2]
    xy = np.zeros((1, 17, 2), dtype=np.float32)
    conf = np.zeros((1, 17), dtype=np.float32)

    lm = res.pose_landmarks.landmark
    for coco_idx, mp_idx in mapping.items():
        p = lm[mp_idx]
        xy[0, coco_idx] = np.array([p.x * w, p.y * h], dtype=np.float32)
        conf[0, coco_idx] = float(p.visibility)

    return xy, conf


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

    def valid(conf: np.ndarray, idx: int) -> bool:
        return conf[idx] >= kpt_conf_thres

    def to_i(pt: np.ndarray) -> tuple[int, int]:
        return int(pt[0]), int(pt[1])

    def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a + b) / 2.0

    def draw_construction_ellipse(
        img: np.ndarray,
        center: np.ndarray,
        major: float,
        minor: float,
        angle_deg: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        cv2.ellipse(
            img,
            to_i(center),
            (max(2, int(major)), max(2, int(minor))),
            angle_deg,
            0,
            360,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def draw_limb_mass(
        img: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        width: int,
        color: tuple[int, int, int],
    ) -> None:
        cv2.line(img, to_i(p1), to_i(p2), color, max(1, width), cv2.LINE_AA)
        cv2.circle(img, to_i(p1), max(2, width // 2), color, 1, cv2.LINE_AA)
        cv2.circle(img, to_i(p2), max(2, width // 2), color, 1, cv2.LINE_AA)

    # 关键点索引
    NOSE = 0
    L_SHO, R_SHO = 5, 6
    L_ELB, R_ELB = 7, 8
    L_WRI, R_WRI = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNE, R_KNE = 13, 14
    L_ANK, R_ANK = 15, 16

    for person_xy, person_conf in zip(keypoints_xy, keypoints_conf):
        # 先画基础骨架（淡灰）
        for i, j in COCO_EDGES:
            if valid(person_conf, i) and valid(person_conf, j):
                cv2.line(
                    canvas,
                    to_i(person_xy[i]),
                    to_i(person_xy[j]),
                    (190, 190, 190),
                    max(1, line_thickness - 1),
                    cv2.LINE_AA,
                )

        # 估算比例（用于身体块面大小）
        body_scale = 60.0
        if valid(person_conf, L_SHO) and valid(person_conf, R_SHO):
            body_scale = float(np.linalg.norm(person_xy[L_SHO] - person_xy[R_SHO]))
        if body_scale < 16:
            body_scale = 16

        # 头部（球体 + 中轴）
        if valid(person_conf, NOSE):
            head_center = person_xy[NOSE].copy()
            head_center[1] -= body_scale * 0.20
            head_r = body_scale * 0.33
            cv2.circle(
                canvas,
                to_i(head_center),
                int(max(4, head_r)),
                (60, 60, 60),
                max(1, line_thickness - 1),
                cv2.LINE_AA,
            )
            cv2.line(
                canvas,
                (int(head_center[0]), int(head_center[1] - head_r)),
                (int(head_center[0]), int(head_center[1] + head_r)),
                (150, 150, 150),
                1,
                cv2.LINE_AA,
            )

        # 躯干：胸腔椭圆 + 骨盆椭圆 + 脊柱线（更接近人体结构草图）
        if valid(person_conf, L_SHO) and valid(person_conf, R_SHO):
            shoulder_mid = midpoint(person_xy[L_SHO], person_xy[R_SHO])
            shoulder_vec = person_xy[R_SHO] - person_xy[L_SHO]
            shoulder_angle = float(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0])))

            rib_center = shoulder_mid + np.array([0.0, body_scale * 0.35], dtype=np.float32)
            draw_construction_ellipse(
                canvas,
                center=rib_center,
                major=body_scale * 0.42,
                minor=body_scale * 0.62,
                angle_deg=shoulder_angle,
                color=(45, 45, 45),
                thickness=max(1, line_thickness - 1),
            )

            if valid(person_conf, L_HIP) and valid(person_conf, R_HIP):
                hip_mid = midpoint(person_xy[L_HIP], person_xy[R_HIP])
                hip_vec = person_xy[R_HIP] - person_xy[L_HIP]
                hip_angle = float(np.degrees(np.arctan2(hip_vec[1], hip_vec[0])))
                pelvis_center = hip_mid
                draw_construction_ellipse(
                    canvas,
                    center=pelvis_center,
                    major=body_scale * 0.45,
                    minor=body_scale * 0.35,
                    angle_deg=hip_angle,
                    color=(45, 45, 45),
                    thickness=max(1, line_thickness - 1),
                )
                cv2.line(
                    canvas,
                    to_i(rib_center),
                    to_i(pelvis_center),
                    (90, 90, 90),
                    max(1, line_thickness - 2),
                    cv2.LINE_AA,
                )

        # 四肢：粗线段 + 关节圆，弱化“火柴人”感
        limb_pairs = (
            (L_SHO, L_ELB),
            (L_ELB, L_WRI),
            (R_SHO, R_ELB),
            (R_ELB, R_WRI),
            (L_HIP, L_KNE),
            (L_KNE, L_ANK),
            (R_HIP, R_KNE),
            (R_KNE, R_ANK),
        )
        upper_w = max(2, int(line_thickness * 2.2))
        lower_w = max(2, int(line_thickness * 1.8))
        for a, b in limb_pairs:
            if valid(person_conf, a) and valid(person_conf, b):
                w = upper_w if a in (L_SHO, R_SHO, L_HIP, R_HIP) else lower_w
                draw_limb_mass(canvas, person_xy[a], person_xy[b], w, (35, 35, 35))

        # 关键点保留为浅灰辅助点（不再红点）
        for idx, pt in enumerate(person_xy):
            if valid(person_conf, idx):
                cv2.circle(canvas, to_i(pt), max(1, point_radius - 1), (130, 130, 130), -1, cv2.LINE_AA)

    return canvas


def extract_pose_image(
    model: YOLO,
    image: np.ndarray,
    conf: float = 0.25,
    kpt_conf: float = 0.4,
    line_thickness: int = 3,
    point_radius: int = 4,
    use_mp_fallback: bool = True,
) -> np.ndarray:
    """输入BGR图像，返回骨架图(BGR)。"""
    result = model.predict(source=image, conf=conf, verbose=False)[0]
    keypoints_xy = None
    keypoints_conf = None
    if result.keypoints is not None and result.keypoints.xy is not None:
        keypoints_xy = result.keypoints.xy.cpu().numpy()
        if result.keypoints.conf is None:
            keypoints_conf = np.ones(keypoints_xy.shape[:2], dtype=np.float32)
        else:
            keypoints_conf = result.keypoints.conf.cpu().numpy()

    # 半身/遮挡图时 YOLO keypoints 可能太少：回退到 MediaPipe
    visible_points = 0
    if keypoints_conf is not None:
        visible_points = int((keypoints_conf[0] >= kpt_conf).sum())
    if (keypoints_xy is None or visible_points < 6) and use_mp_fallback:
        mp_xy, mp_conf = mp_pose_to_coco17(image)
        if mp_xy is not None:
            keypoints_xy, keypoints_conf = mp_xy, mp_conf

    if keypoints_xy is None or keypoints_conf is None:
        return np.full_like(image, 255)

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
    use_mp_fallback: bool,
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
        use_mp_fallback=use_mp_fallback,
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
                use_mp_fallback=not args.no_mp_fallback,
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
            use_mp_fallback=not args.no_mp_fallback,
        )
        print(f"[OK] {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
