import os, json, string
import cv2
import numpy as np
from ultralytics import YOLO
import torch


# ============ 固定配置 ============
WEIGHTS = "/app/input/res/your_program/cv_models/best.pt"
CONF = 0.6
DEVICE = "cuda"
OUT_IMAGE_DIR = "/app/input/res/your_program/marked_images"
OUT_JSON_DIR = "/app/input/res/your_program/marked_json"

# ============ 模型加载优化 ============
model = None


def get_model():
    """全局单例模型，避免重复加载"""
    global model
    if model is None:
        print("正在加载YOLO模型...")
        model = YOLO(WEIGHTS)
    return model


def release_model():
    """释放 YOLO 模型并清理显存"""
    global model
    if model is not None:
        print("释放 YOLO 模型和显存...")
        del model
        model = None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# =================================

def sort_indices_by_geometry(boxes):
    # y1 升序；同行内 x1 升序
    return sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))


def assign_names(boxes):
    # A,B,C,... 超 26 -> AA, AB, ...
    letters = list(string.ascii_uppercase)
    order = sort_indices_by_geometry(boxes)
    name_of = {}
    for k, i in enumerate(order):
        if k < len(letters):
            name_of[i] = letters[k]
        else:
            name_of[i] = letters[(k // 26) - 1] + letters[k % 26]
    return name_of


def run_yolo(image_path):
    # 使用全局模型，避免重复加载
    model = get_model()
    r = model.predict(image_path, conf=CONF, device=DEVICE, verbose=False)[0]
    boxes_xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else np.zeros((0, 4))

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")
    h, w = img.shape[:2]

    boxes = []
    for x1, y1, x2, y2 in boxes_xyxy:
        xi1 = int(round(max(0, x1)))
        yi1 = int(round(max(0, y1)))
        xi2 = int(round(min(w - 1, x2)))
        yi2 = int(round(min(h - 1, y2)))
        if xi2 > xi1 and yi2 > yi1:
            boxes.append([xi1, yi1, xi2, yi2])
    return boxes, img


def process_image(image_path, output_image_dir, output_json_dir):
    """
    处理单张图片：生成标记图片和对应的JSON文件
    """
    try:
        # 1) YOLO 检测
        boxes, img = run_yolo(image_path)

        if not boxes:
            print(f"在图片 {os.path.basename(image_path)} 中未检测到组件")
            return

        # 2) 为组件命名
        name_of = assign_names(boxes)

        # 3) 生成JSON数据
        items = []
        for i, pos in enumerate(boxes):
            items.append({
                "Component": name_of[i],
                "Pos": pos,
                "I_O": {"input": 0, "output": 0},
                "Connection": {
                    "input": [],
                    "output": []
                }
            })
        items.sort(key=lambda d: d["Component"])  # 按字母序

        # 4) 保存JSON文件
        base = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(output_json_dir, f"{base}.json")

        # 创建外层包含task1字段的JSON结构
        output_data = {
            "task1": items
        }

        # 使用 utf-8 编码
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

            # 5) 在图片上绘制标记
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                # 绘制绿色边框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 获取组件名称
                component_name = name_of[i]

                # 计算边框的宽度和高度
                box_width = x2 - x1
                box_height = y2 - y1

                # 设置安全边距（确保文字不会碰到边框）
                margin = max(3, int(min(box_width, box_height) * 0.08))

                # 计算可用的文本区域（进一步缩小确保安全）
                text_area_width = box_width - 3 * margin  # 额外多减一个margin确保安全
                text_area_height = box_height - 3 * margin

                # 如果文本区域太小，跳过文字绘制
                if text_area_width <= 0 or text_area_height <= 0:
                    continue

                # 自适应计算字体大小
                font_scale = 0.1  # 初始字体大小
                thickness = 4
                font = cv2.FONT_HERSHEY_SIMPLEX

                # 通过迭代找到合适的字体大小
                max_scale = font_scale
                for scale in range(1, 200):  # 增加迭代范围
                    test_scale = scale * 0.05  # 减小步长以获得更精确的适配
                    text_size = cv2.getTextSize(component_name, font, test_scale, thickness)[0]
                    text_width, text_height = text_size

                    # 检查文本是否完全在可用区域内（包含安全余量）
                    if text_width <= text_area_width and text_height <= text_area_height:
                        max_scale = test_scale
                    else:
                        break

                font_scale = max_scale

                # 重新计算最终文本尺寸
                text_size = cv2.getTextSize(component_name, font, font_scale, thickness)[0]
                text_width, text_height = text_size

                # 计算文本位置（严格居中，确保在边框内）
                text_x = x1 + (box_width - text_width) // 2
                text_y = y1 + (box_height + text_height) // 2 - int(text_height * 0.1)  # 微调垂直位置

                # 检查文本位置是否在边框内
                if (text_x < x1 + margin or
                        text_x + text_width > x2 - margin or
                        text_y - text_height < y1 + margin or
                        text_y > y2 - margin):
                    # 如果超出边界，调整到安全位置
                    text_x = max(x1 + margin, min(text_x, x2 - margin - text_width))
                    text_y = min(y2 - margin, max(text_y, y1 + margin + text_height))

                # 绘制文本背景（白色半透明）- 确保在边框内
                bg_margin = 2
                bg_x1 = max(x1 + 1, text_x - bg_margin)
                bg_y1 = max(y1 + 1, text_y - text_height - bg_margin)
                bg_x2 = min(x2 - 1, text_x + text_width + bg_margin)
                bg_y2 = min(y2 - 1, text_y + bg_margin)

                # 创建白色背景
                overlay = img.copy()
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)

                # 使用透明度混合
                alpha = 1  # 背景透明度
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                # 绘制红色文本
                cv2.putText(img, component_name, (text_x, text_y),
                            font, font_scale, (0, 0, 255), thickness)

        # ====== 新增：等比例缩放到长边 1024 ======
        max_side = 1024
        h, w = img.shape[:2]
        scale = max_side / max(h, w)

        new_w = int(w * scale)
        new_h = int(h * scale)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # ============================================

        # 6) 保存图片
        image_output_path = os.path.join(output_image_dir, os.path.basename(image_path))
        cv2.imwrite(image_output_path, img)

        print(f"[OK] 已处理图片 {os.path.basename(image_path)}:")
        print(f"     检测到 {len(boxes)} 个组件")
        print(f"     标记图片保存到：{image_output_path}")
        print(f"     JSON文件保存到：{json_path}")

    except Exception as e:
        print(f"处理图片 {image_path} 时出错：{e}")


def getJSON(image_pa):
    # 获取输入文件夹路径
    input_dir = image_pa

    if not input_dir or not os.path.isdir(input_dir):
        print("路径无效或文件夹不存在，程序结束。")
        return

    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # 遍历文件夹中的所有图片文件
    image_files = []
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(fmt) for fmt in supported_formats):
            image_files.append(os.path.join(input_dir, filename))

    if not image_files:
        print("在指定文件夹中未找到支持的图片文件（jpg, jpeg, png, bmp, tiff）")
        return

    print(f"找到 {len(image_files)} 张图片，开始处理...")

    # 批量处理所有图片
    for i, image_path in enumerate(image_files, 1):
        print(f"\n处理进度：{i}/{len(image_files)}")
        process_image(image_path, OUT_IMAGE_DIR, OUT_JSON_DIR)

    release_model()

    print(f"\n批量处理完成！")
    print(f"所有标记后的图片已保存到：{OUT_IMAGE_DIR}")
    print(f"所有JSON文件已保存到：{OUT_JSON_DIR}")