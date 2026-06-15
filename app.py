import traceback
import cv2
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from recognize import FaceRecognizer


recognizer = None


def get_recognizer():
    """
    延迟加载识别器。
    第一次点击识别时才加载模型，避免页面启动阶段直接卡死。
    """
    global recognizer
    if recognizer is None:
        recognizer = FaceRecognizer()
    return recognizer


def get_chinese_font(size=24):
    """
    加载 Windows 中文字体，解决 OpenCV putText 中文乱码问题。
    """
    font_paths = [
        r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
        r"C:\Windows\Fonts\simhei.ttf",    # 黑体
        r"C:\Windows\Fonts\simsun.ttc",    # 宋体
        r"C:\Windows\Fonts\simkai.ttf",    # 楷体
    ]

    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            continue

    return ImageFont.load_default()


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def normalize_bbox(bbox):
    """
    bbox 统一处理成 [x, y, w, h]
    """
    if bbox is None:
        return None

    if len(bbox) < 4:
        return None

    try:
        x, y, w, h = [int(v) for v in bbox[:4]]
        return x, y, w, h
    except Exception:
        return None


def draw_results_with_chinese(image_rgb, results):
    """
    使用 PIL 在图片上绘制人脸框和中文标签。
    不使用 cv2.putText，因为 cv2.putText 不支持中文。
    """
    pil_img = Image.fromarray(image_rgb.astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)

    font = get_chinese_font(size=24)

    for item in results:
        bbox = normalize_bbox(item.get("bbox"))
        if bbox is None:
            continue

        x, y, w, h = bbox

        identity_id = item.get("identity_id", "unknown")
        name = item.get("name", "")
        similarity = safe_float(item.get("similarity", 0.0))

        if not name:
            name = "未知人物"

        if identity_id == "unknown":
            label = f"unknown 未知人物 {similarity:.4f}"
            box_color = (255, 70, 70)
            text_color = (255, 70, 70)
        else:
            label = f"{identity_id} {name} {similarity:.4f}"
            box_color = (0, 220, 80)
            text_color = (0, 220, 80)

        # 画人脸框
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline=box_color,
            width=4
        )

        # 文字位置
        text_x = x
        text_y = y - 32
        if text_y < 0:
            text_y = y + 4

        # 画文字黑色背景，增强可读性
        try:
            text_bbox = draw.textbbox((text_x, text_y), label, font=font)
            bg_left = text_bbox[0] - 4
            bg_top = text_bbox[1] - 2
            bg_right = text_bbox[2] + 4
            bg_bottom = text_bbox[3] + 2

            draw.rectangle(
                [(bg_left, bg_top), (bg_right, bg_bottom)],
                fill=(0, 0, 0)
            )
        except Exception:
            pass

        # 写中文标签
        draw.text(
            (text_x, text_y),
            label,
            font=font,
            fill=text_color
        )

    return np.array(pil_img)


def recognize_face(image_rgb):
    """
    Gradio 主识别函数：
    上传图片 -> 调用成员2后端识别接口 -> 绘制中文可视化结果 -> 输出识别信息
    """
    if image_rgb is None:
        return None, "请先上传图片。"

    try:
        # Gradio 输入是 RGB，成员2后端一般使用 OpenCV BGR
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        model = get_recognizer()
        results = model.recognize_image(image_bgr)

        # 不再调用成员2的 draw_results，避免 OpenCV 中文乱码
        rendered_rgb = draw_results_with_chinese(image_rgb.copy(), results)

        lines = []
        lines.append(f"检测到人脸数量：{len(results)}")

        if len(results) == 0:
            lines.append("未检测到人脸。")

        for i, item in enumerate(results, start=1):
            identity_id = item.get("identity_id", "unknown")
            name = item.get("name", "")

            if not name:
                name = "未知人物"

            similarity = safe_float(item.get("similarity", 0.0))
            detection_score = safe_float(item.get("detection_score", 0.0))
            bbox = item.get("bbox", "")

            lines.append("")
            lines.append(f"第 {i} 张人脸：")
            lines.append(f"身份ID：{identity_id}")
            lines.append(f"姓名：{name}")
            lines.append(f"相似度：{similarity:.4f}")
            lines.append(f"检测置信度：{detection_score:.4f}")
            lines.append(f"人脸框 bbox：{bbox}")

        return rendered_rgb, "\n".join(lines)

    except Exception:
        error_info = traceback.format_exc()
        return image_rgb, "程序运行出错：\n\n" + error_info


with gr.Blocks(title="AI 人脸识别系统") as demo:
    gr.Markdown("# AI 人脸识别系统")
    gr.Markdown(
        "上传图片后，系统会自动完成：人脸检测、人脸识别、人脸框绘制、身份 ID / 姓名 / 相似度展示。"
    )

    with gr.Row():
        input_image = gr.Image(
            label="上传待识别图片",
            type="numpy"
        )

        output_image = gr.Image(
            label="识别结果可视化",
            type="numpy"
        )

    start_button = gr.Button("开始识别")

    output_text = gr.Textbox(
        label="识别结果信息",
        lines=14
    )

    start_button.click(
        fn=recognize_face,
        inputs=input_image,
        outputs=[output_image, output_text]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        show_api=False
    )