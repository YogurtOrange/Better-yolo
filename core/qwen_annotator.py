# -*- coding: utf-8 -*-
"""
Qwen-VL AI 自动标注核心模块
移植自独立脚本版 ai_yolo_annotator_gui.py 的纯逻辑部分，
适配 Better-YOLO 的数据库与项目体系。

功能:
  1. 单图单请求 Qwen-VL 检测 (整数像素坐标 + bbox_2d/label JSON)
  2. 坐标空间自动判定 (0~1 / 0~1000 / 像素)
  3. 多轮检测 IoU 聚类合并 (减少框抖动)
  4. 二次放大精修 (zoom-in re-grounding, 迭代收敛)
  5. 提示词内置类别表 + 严格类别校验 (防止AI自创类别)
  6. few-shot 示例图片
  7. AI 生成/迭代任务提示词
  8. 断点续标 (项目目录 ai_progress.json)
  9. 结果直接写入数据库 (annotations 表)，在"标注"页面直接显示

本模块不依赖任何 Qt, 可在 QThread 中直接调用。
"""

import os
import re
import json
import time
import base64
import random
import threading
import concurrent.futures

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from openai import OpenAI

from models.database import db


# ============================================================
# 默认配置
# ============================================================

DEFAULT_MODEL = "qwen3-vl-plus"

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 每张图重复检测次数（>1时取平均，减少框抖动）
DEFAULT_DETECT_ROUNDS = 1

# 二次放大精修默认关闭(用户可在界面上勾选开启)
DEFAULT_ZOOM_REFINE = False

# 二次放大精修迭代轮数(第1轮大视野纠偏，后续轮精确定位)
DEFAULT_ZOOM_ITERS = 2

# 上传最大像素(单位:万)
DEFAULT_UPLOAD_MEGAPIXELS = 200

# API重试次数
DEFAULT_MAX_RETRY = 5

# 并发请求数
DEFAULT_MAX_WORKERS = 10

# 支持的图片格式
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


class StopRequested(Exception):
    """用户请求停止"""
    pass


def make_client(api_key, base_url=BASE_URL):
    """创建 OpenAI 兼容客户端"""
    if not api_key:
        raise RuntimeError(
            "没有找到 API Key，请在设置中配置或执行: "
            "export DASHSCOPE_API_KEY='你的API_KEY'"
        )
    return OpenAI(api_key=api_key, base_url=base_url or BASE_URL)


# ============================================================
# API请求自动重试
# ============================================================

def call_qwen(client, model, messages, max_retry, log, stop_event=None):

    for retry in range(max_retry):

        if stop_event is not None and stop_event.is_set():
            raise StopRequested()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                timeout=120
            )

            usage = getattr(response, "usage", None)
            if usage:
                cached = getattr(
                    getattr(usage, "prompt_tokens_details", None),
                    "cached_tokens", None
                )
                log(
                    f"token用量: prompt={usage.prompt_tokens} "
                    f"completion={usage.completion_tokens} "
                    f"cached={cached}"
                )

            return response

        except StopRequested:
            raise

        except Exception as e:
            log(f"API失败 ({retry+1}/{max_retry}): {e}")

            if retry < max_retry - 1:
                sleep_time = 2 ** retry
                log(f"{sleep_time}s后重试...")
                # 分段睡眠，便于及时响应停止
                for _ in range(sleep_time):
                    if stop_event is not None and stop_event.is_set():
                        raise StopRequested()
                    time.sleep(1)
            else:
                raise e


# ============================================================
# 图片压缩与编码
# ============================================================

def compress_image(image_path, megapixels):
    """压缩图片，返回原图/压缩图尺寸信息用于bbox恢复"""
    img = Image.open(image_path).convert("RGB")

    original_width, original_height = img.size

    target_pixels = megapixels * 10000

    scale = (
        target_pixels / (original_width * original_height)
    ) ** 0.5

    # 不放大
    scale = min(scale, 1.0)

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    return {
        "image": img,
        "original_size": (original_width, original_height),
        "resize_size": (new_width, new_height)
    }


def encode_image(image_path, megapixels):
    """压缩图片转base64 data url"""
    data = compress_image(image_path, megapixels)

    buffer = BytesIO()
    data["image"].save(buffer, format="JPEG", quality=90)

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    data["url"] = "data:image/jpeg;base64," + image_base64

    return data


def encode_pil_image(img, quality=90):
    """PIL图片转base64 data url"""
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(buffer.getvalue()).decode("utf-8")
    )


# ============================================================
# Qwen目标检测 (单图单请求)
# ============================================================

SYSTEM_PROMPT = """
你是一个专业的YOLO数据集标注AI。

用户消息中会给出标注任务(含类别表)和一张待标注图片。

要求:

1) 以JSON数组输出，每个元素格式为
{"bbox_2d": [x1, y1, x2, y2], "label": "类别名"}。
只输出JSON，不要输出任何其它文字。

2) bbox坐标必须是【整数像素值】(不是0~1的比例，也
不是0~1000): 相对你收到的这张图的实际分辨率，x1,y1
为左上角，x2,y2为右下角，最小0，最大为图像的宽/高。
始终保证 x1<x2 且 y1<y2。

3) 图中通常有多个物体，每一个都要单独输出一条，
同一类出现多次就输出多条。

4) 框要紧贴物体本身的可见轮廓，只框物体本身，绝不
包含周围的背景留白: 阴影、地面、桌面、支架、底座，
以及目标所附着或印刷的面板/靶板/纸张都不要框进来。
在较大和较小的框之间犹豫时，选择较小的那个。

5) 物体互相遮挡或被图片边缘裁切时，框其可见部分的
完整外轮廓，框边缘必须精确到达目标被裁切处的图片
边缘。

6) label 必须严格使用类别表中冒号前的标签名，逐字
一致，包括语言和用字，不得翻译、不得改名、不得合并；
特别注意: 标签名即使是纯数字编号(如"001")也必须
原样输出，绝不能用描述里提到的物体名称(如"老虎")
当作label；不要发明类别表之外的任何类别。

7) 不确定是哪一类的物体，宁可不输出，也不要猜一个
类别名。

8) 图中没有任何目标就输出空数组 []。

9) 标记为 "Reference example" 的图片只用于演示标注
风格(类别、框的位置松紧)，不要在示例图上找目标，
更不要把它们输出到结果里。
"""


# 稳健解析: 剥掉```json围栏，取最外层[...]，
# 兼容 bbox_2d/bbox/box 与 label/category/class_name/name
def parse_boxes(text):

    text = (text or "").strip()

    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)

    if m:
        text = m.group(1).strip()

    s, e = text.find("["), text.rfind("]")

    if s == -1 or e == -1:
        return []

    try:
        data = json.loads(text[s:e + 1])
    except Exception:
        return []

    out = []

    for d in data:
        if not isinstance(d, dict):
            continue

        bb = d.get("bbox_2d") or d.get("bbox") or d.get("box")

        label = (
            d.get("label")
            or d.get("category")
            or d.get("class_name")
            or d.get("name")
        )

        if bb and label and len(bb) == 4:
            try:
                out.append((str(label), [float(v) for v in bb]))
            except Exception:
                continue

    return out


# 坐标空间自动判定: 看这张图所有坐标的最大值 ->
# 0~1比例 / 0~1000 / 像素，返回 (div_w, div_h)
def guess_coord_div(boxes, ref_w, ref_h, force_pixel=False):

    if force_pixel:
        return float(ref_w), float(ref_h)

    vals = []

    for _, bb in boxes:
        vals += [abs(v) for v in bb]

    mx = max(vals) if vals else 0

    if mx <= 1.5:
        return 1.0, 1.0

    if mx <= 1000:
        return 1000.0, 1000.0

    return float(ref_w), float(ref_h)


# qwen2-vl / qwen2.5-vl 系列输出绝对像素坐标，
# 其余(qwen-vl老一代、qwen3-vl等)输出0~1000归一化坐标。
# 裁剪图通常小于1000px，仅靠数值启发式无法区分，
# 需要按模型家族辅助判定。
def model_prefers_abs_pixel(model):

    return bool(re.search(r"qwen2(\.5)?-vl", model or "", re.I))


# 模型返回坐标 -> 原图像素坐标
def boxes_to_original_px(boxes, info, model=""):

    rw, rh = info["resize_size"]
    ow, oh = info["original_size"]

    sx, sy = ow / rw, oh / rh

    div_w, div_h = guess_coord_div(
        boxes, rw, rh,
        force_pixel=model_prefers_abs_pixel(model)
    )

    out = []

    for label, bb in boxes:

        x1, y1, x2, y2 = bb

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        px1 = x1 / div_w * rw * sx
        py1 = y1 / div_h * rh * sy
        px2 = x2 / div_w * rw * sx
        py2 = y2 / div_h * rh * sy

        px1 = max(0.0, min(px1, ow))
        px2 = max(0.0, min(px2, ow))
        py1 = max(0.0, min(py1, oh))
        py2 = max(0.0, min(py2, oh))

        out.append((label, [px1, py1, px2, py2]))

    return out


# 单图检测: 返回(该图objects列表, 图片压缩info)
# objects: [{"class_name","bbox"(原图像素),"confidence"}]
def detect_single(client, model, image_path, task_text,
                  example_parts, megapixels, max_retry,
                  log, stop_event):

    user_content = [{"type": "text", "text": task_text}]

    # 示例图片（few-shot，不参与结果；已预编码，直接复用）
    for part in example_parts:
        user_content.append(part)

    info = encode_image(image_path, megapixels)

    user_content.append({"type": "text", "text": "待标注图片:"})
    user_content.append({
        "type": "image_url",
        "image_url": {"url": info["url"]}
    })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    response = call_qwen(
        client, model, messages, max_retry, log, stop_event
    )

    boxes = parse_boxes(response.choices[0].message.content)

    objects = []

    for label, px in boxes_to_original_px(boxes, info, model):
        objects.append({
            "class_name": label,
            "bbox": [round(v, 1) for v in px],
            "confidence": 1.0
        })

    return objects, info


# ============================================================
# 多轮检测结果合并(单张图)
#
# Qwen grounding 存在随机抖动，同一张图检测多次，
# 用 IoU 匹配同一目标的框，坐标取平均可显著减小偏移。
# 出现次数不足半数的框视为误检丢弃，类别取多数。
# ============================================================

def bbox_iou(a, b):

    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    inter = ix * iy

    if inter <= 0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (area_a + area_b - inter)


def merge_round_objects(obj_lists, rounds):

    if rounds == 1 or len(obj_lists) == 1:
        return list(obj_lists[0])

    # 保留阈值: 出现次数 >= 过半轮数
    min_count = (rounds + 1) // 2

    # 聚类: IoU>=0.5 视为同一目标
    clusters = []

    for objs in obj_lists:

        for obj in objs:

            bbox = [float(x) for x in obj["bbox"]]

            best = None
            best_iou = 0.5

            for cluster in clusters:

                mean_box = [
                    sum(b[i] for b in cluster["boxes"])
                    / len(cluster["boxes"])
                    for i in range(4)
                ]

                v = bbox_iou(bbox, mean_box)

                if v > best_iou:
                    best = cluster
                    best_iou = v

            if best is None:
                clusters.append({
                    "boxes": [bbox],
                    "classes": [obj["class_name"]],
                    "confs": [obj.get("confidence", 1.0)]
                })
            else:
                best["boxes"].append(bbox)
                best["classes"].append(obj["class_name"])
                best["confs"].append(
                    obj.get("confidence", 1.0)
                )

    objects = []

    for cluster in clusters:

        n = len(cluster["boxes"])

        if n < min_count:
            continue

        avg_bbox = [
            sum(b[i] for b in cluster["boxes"]) / n
            for i in range(4)
        ]

        cls = max(
            set(cluster["classes"]),
            key=cluster["classes"].count
        )

        objects.append({
            "class_name": cls,
            "bbox": [round(x, 1) for x in avg_bbox],
            "confidence": round(sum(cluster["confs"]) / n, 3)
        })

    return objects


# ============================================================
# 二次放大精修 (zoom-in re-grounding)
#
# 对每个检出目标按框外扩裁剪出局部小图并放大到高分辨率，
# 让AI在局部图上重新输出贴合框，再映射回原图坐标。
# 第1轮大视野裁剪(容忍粗检大幅偏移)，后续轮小视野
# 裁剪(追求贴合精度)。框收敛后提前结束。
# 单目标失败保留原框，不中断流程
# ============================================================

REFINE_SYSTEM_PROMPT = """你是一个精准的视觉定位助手。

你会收到若干张放大裁剪的局部图。每张局部图中恰好
包含一个目标物体，其类别在局部图前的"目标类别"文字
中给出。

你的任务: 输出该目标在局部图中的紧贴边界框。

规则：

1. 只返回JSON，不要附加任何额外文字。

2. bbox_2d坐标为【整数像素值】，相对该局部图本身的
实际分辨率(不是原始大图)，格式[x1,y1,x2,y2]，
x1<x2，y1<y2。

3. 框边缘必须贴住目标物体的最外可见轮廓，
不得包含背景留白。

4. 排除阴影、地面、板材、胶带以及目标所附着或
印刷的任何东西。

5. 如果目标被局部图边界裁切，框边缘必须精确延伸到
该边界，完整包含可见部分。

6. 如果局部图中找不到目标，该裁剪输出bbox_2d=[0,0,0,0]。

7. "目标类别"名称只是一个标签。目标的精确定义和
框的松紧以用户消息中的任务上下文为准。例如任务
要求框外切于包围圆圈、或包含被裁切目标的完整可见
部分时，必须遵循任务要求，即使类名字面含义暗示
不同的范围。

JSON格式:
{"results":[{"crop":"crop_1","bbox_2d":[x1,y1,x2,y2]}]}"""


# 按bbox外扩裁剪局部图
# margin_ratio: 每侧外扩量占bbox边长的比例
# min_size: 裁剪图过小时放大到该尺寸(让AI看得更清)
# 返回 (crop图, 裁剪原点, 放大倍数)
def build_crop(img, bbox_px, margin_ratio=1.2, min_size=768):

    W, H = img.size

    x1, y1, x2, y2 = bbox_px

    bw = max(4.0, x2 - x1)
    bh = max(4.0, y2 - y1)

    mx = bw * margin_ratio / 2
    my = bh * margin_ratio / 2

    cx1 = max(0, int(x1 - mx))
    cy1 = max(0, int(y1 - my))
    cx2 = min(W, int(x2 + mx))
    cy2 = min(H, int(y2 + my))

    # 极端情况兜底: 保证crop至少有几个像素
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        cx1 = max(0, int(x1) - 4)
        cy1 = max(0, int(y1) - 4)
        cx2 = min(W, int(x2) + 4)
        cy2 = min(H, int(y2) + 4)

    crop = img.crop((cx1, cy1, cx2, cy2))

    scale = 1.0

    if crop.width < min_size or crop.height < min_size:
        scale = max(
            min_size / crop.width,
            min_size / crop.height
        )
        crop = crop.resize(
            (int(crop.width * scale) + 1,
             int(crop.height * scale) + 1),
            Image.LANCZOS
        )

    crop.load()

    return crop, (cx1, cy1), scale


def extract_json_object(text):
    """稳健提取最外层JSON对象(剥掉```json围栏等杂质)"""
    text = (text or "").strip()

    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)

    if m:
        text = m.group(1).strip()

    s, e = text.find("{"), text.rfind("}")

    if s == -1 or e == -1:
        return None

    try:
        return json.loads(text[s:e + 1])
    except Exception:
        return None


def zoom_refine_objects(client, model, image_path,
                        objects, max_retry,
                        log, stop_event, iters=2,
                        task_text=""):

    if not objects:
        return

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        W, H = img.size

        # 收敛阈值: 最大边长的0.5%
        shift_th = max(W, H) * 0.005

        for it in range(max(1, iters)):

            # 首轮大视野纠偏，之后小视野精确定位
            margin = 1.5 if it == 0 else 0.5

            # 为每个目标生成裁剪图
            crops = []

            for obj in objects:
                crop, origin, scale = build_crop(
                    img, obj["bbox"], margin_ratio=margin
                )
                crops.append((obj, crop, origin, scale))

            messages = [{
                "role": "system",
                "content": REFINE_SYSTEM_PROMPT
            }]

            refine_text = (
                "下面的裁剪图是检出目标区域的放大视图。"
                "对每张裁剪图，输出所述目标类别的紧贴边界框，"
                "坐标为相对该裁剪图分辨率的整数像素值。"
                "JSON中crop字段必须逐字复制裁剪编号"
                "(如crop_1)。"
            )

            # 任务上下文决定目标的精确定义和框的松紧，
            # 避免AI只按类名字面意思框住符号本身
            if task_text:
                refine_text += (
                    "\n\n任务上下文(它定义了目标的精确范围和框的"
                    "松紧，必须严格遵循):\n" + task_text
                )

            content = [{"type": "text", "text": refine_text}]

            for i, (obj, crop, origin, scale) in enumerate(crops):
                content.append({
                    "type": "text",
                    "text": (
                        f"裁剪编号: crop_{i + 1}\n"
                        f"目标类别: {obj['class_name']}"
                    )
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": encode_pil_image(crop)}
                })

            messages.append({"role": "user", "content": content})

            response = call_qwen(
                client, model, messages, max_retry, log, stop_event
            )

            raw = response.choices[0].message.content
            data = extract_json_object(raw)

            if not data:
                log("[警告] 精修响应解析失败，保留原框")
                return

            refined_map = {}

            for item in data.get("results", []):
                refined_map[item.get("crop", "")] = (
                    item.get("bbox_2d") or item.get("bbox")
                )

            # 就地更新bbox: 裁剪图像素坐标 -> 原图像素坐标
            max_shift = 0.0

            for i, (obj, crop, origin, scale) in enumerate(crops):

                bbox = refined_map.get(f"crop_{i + 1}")

                if not bbox or len(bbox) != 4:
                    log(f"[警告] crop_{i + 1} 无精修结果，保留原框")
                    continue

                try:
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                except (TypeError, ValueError):
                    log(f"[警告] crop_{i + 1} 精修结果异常，保留原框")
                    continue

                # [0,0,0,0]表示AI在crop中找不到目标，保留原框
                if x2 <= x1 or y2 <= y1:
                    log(f"[警告] crop_{i + 1} 未找到目标，保留原框")
                    continue

                ox, oy = origin

                # 坐标空间自动判定(与首轮检测一致):
                # 部分模型(qwen-vl-max等)无视像素指令，
                # 对裁剪图也返回0~1000归一化坐标
                cw, ch = crop.size
                div_w, div_h = guess_coord_div(
                    [("", bbox)], cw, ch,
                    force_pixel=model_prefers_abs_pixel(model)
                )

                fx = cw / div_w / scale
                fy = ch / div_h / scale

                # 裁剪图可能被放大过(scale>1)，除回去
                px1 = ox + x1 * fx
                py1 = oy + y1 * fy
                px2 = ox + x2 * fx
                py2 = oy + y2 * fy

                px1 = max(0.0, min(px1, W))
                px2 = max(0.0, min(px2, W))
                py1 = max(0.0, min(py1, H))
                py2 = max(0.0, min(py2, H))

                if px2 - px1 < 2 or py2 - py1 < 2:
                    log(f"[警告] crop_{i + 1} 精修框过小，保留原框")
                    continue

                new_bbox = [
                    round(px1, 1), round(py1, 1),
                    round(px2, 1), round(py2, 1),
                ]

                # 记录本轮最大变化量(原图像素)
                shift = max(
                    abs(a - b)
                    for a, b in zip(new_bbox, obj["bbox"])
                )
                max_shift = max(max_shift, shift)

                obj["bbox"] = new_bbox

            log(
                f"  二次精修第{it + 1}/{iters}轮: "
                f"最大偏移{max_shift:.1f}px"
            )

            # 所有框已收敛，无需继续
            if max_shift < shift_th:
                break


# ============================================================
# AI生成标注任务提示词
# ============================================================

PROMPT_GEN_SYSTEM_PROMPT = """
你是一个YOLO数据集自动标注流水线的提示词工程助手。
流水线会把任务提示词发给视觉语言模型，模型严格
按照提示词在图片上画检测框。

你的任务：为标注模型编写任务提示词(中文)，
使标注结果符合用户意图。

规则：

1. 只返回JSON，格式：
{"prompt": "...", "negative": "..."}

negative 是"负样本排除"内容: 画面里存在但不是标注
目标、必须忽略的东西(顿号分隔，如"靶板、地面、阴影、
支架")。若提供了对比图组，negative 必须把其中AI误框
过的所有非目标物体都列进去；没有需要排除的东西时
可为空字符串。

2. 提示词必须用中文书写。

3. 提示词中的类别表必须严格使用以下格式，
每个类别一行：

标签名：该标签所代表目标的描述

- 标签名直接决定数据集的类别名。用户消息中已有
  提示词/类别表时，其中的标签名(冒号前的部分)
  必须逐字沿用，一行对应一行，严禁增删、改名、
  合并或调整顺序——即使标签名是纯数字编号(如001、
  002)也必须原样保留，编号就是标签名本身，不是
  序号，不要改写成描述中提到的物体名称；
- 只有当用户没有提供任何标签名时，才自行命名，
  此时应使用能说明目标是什么的有意义名称；
- 冒号后的描述是该标签的归类条件，必须具体、有
  区分度，至少包含:
  (1)目标是什么/印有什么图案;
  (2)2~3个关键视觉特征(颜色、纹理、形状、姿态、
  相对大小等)，这是标注模型识别目标的依据;
  (3)框的贴边要求(框紧贴什么、排除什么);
  (4)如有易混淆物体，写明区别;
- 描述写成连贯的一句话，放在一对全角括号内，
  并以单引号复述一遍标签名所代表的目标名称开头。
  每行的格式必须是: 标签名：描述 (标签名、全角
  冒号、描述三者缺一不可)，例如:
  001：装甲车（'装甲车'指靶纸上印有的橙黑条纹虎形
  图案，四肢着地、体型明显大于画面中其他图案，框
  紧贴其身体可见外轮廓，不含靶纸和网格线）;
- 每个类别一行，行与行之间不要有空行；
- 类别列表之后必须另起一行写:
  若目标被部分遮挡，仅框出可见部分的完整外轮廓。
- 如有其他排除要求(如排除靶板/地面/阴影/支架等)，
  再另起一行补充。

示例格式(假设用户的标签名是001和002):

001：装甲车（'装甲车'指靶纸上印有的轮式装甲车图案，深绿色车身、有多组负重轮、炮管前伸，框紧贴装甲车图案可见外轮廓，不含靶纸和网格线）
002：桥梁（'桥梁'指靶纸上印有的灰色钢架桥图案，由交叉桁架构成、横跨画面，框紧贴全部钢架的外轮廓，不含靶纸）
若目标被部分遮挡，仅框出可见部分的完整外轮廓。

4. 从"参考示例"图片中学习标注意图: 用户手画的框
定义了目标是什么、框要贴多紧。类别表中每个标签
的描述必须与示例图中该标签标注的目标类型对应；
标签名的数量和名称仍以用户消息中已有的为准，
不得增删。

5. 类别表的标签名有且只有两个来源: 用户消息中
给出的标签名，或参考示例图片上手写的标签文字。
必须逐字沿用，严禁自行发明示例和用户消息中都不
存在的类别名，严禁根据图片内容臆造中文名(如
"香蕉""水杯"之类)。宁缺勿滥: 不确定标签名的
目标类别不要写进类别表。

6. 提示词中不要提及JSON、坐标尺度或输出格式等细节，
流水线会自行处理。

7. 若提供了"对比图组"(每组两张: 第1张红框=AI上一轮
的标注结果，第2张绿框=用户人工修正后的正确答案)，
必须逐组对比，找出AI犯的错误模式(把背景/干扰物框成
目标、类别认错、框太松/太紧、漏检等)，据此改进prompt
的类别描述与排除要求，并把AI误框过的非目标物体写进
negative。红框中正确的部分保留，错误的部分以绿框为准。
"""


def generate_task_prompt(model, sample_images, example_images,
                         previous_prompt,
                         feedback, megapixels, max_retry,
                         log, stop_event=None,
                         api_key=None, base_url=BASE_URL,
                         extra_note="", comparison_pairs=None):

    client = make_client(api_key, base_url)

    guide = (
        "请为标注模型编写任务提示词。\n"
        "本次请求中的图片角色:\n"
        "- \"采样图片\": 待标注的原始图片，展示场景长什么样。\n"
        "- \"参考示例\": 用户手绘的标准答案。其中的框和标签"
        "定义了目标是什么、框要贴多紧。\n"
    )

    comparison_pairs = comparison_pairs or []

    if comparison_pairs:
        guide += (
            "- \"对比图组\": 每组两张，第1张红框=AI上一轮的标注"
            "结果(含错误)，第2张绿框=用户人工修正后的正确答案。\n"
        )

    if extra_note:
        guide += "\n" + extra_note + "\n"

    if previous_prompt:
        guide += (
            "\n上一轮提示词(请重写并改进):\n"
            + previous_prompt + "\n"
        )

    if feedback:
        guide += (
            "\n用户反馈(请修正这些问题):\n"
            + feedback + "\n"
        )

    content = [{"type": "text", "text": guide}]

    for path in sample_images:
        info = encode_image(path, megapixels)
        content.append({
            "type": "text",
            "text": "采样图片: " + Path(path).name
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": info["url"]}
        })

    for path in example_images:
        info = encode_image(path, megapixels)
        content.append({
            "type": "text",
            "text": "参考示例(用户手绘标准答案，不要计入任何结果)"
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": info["url"]}
        })

    # 对比图组: AI原始结果(红) vs 用户修正结果(绿)
    for idx, pair in enumerate(comparison_pairs, 1):
        name = pair.get("filename", f"组{idx}")

        if pair.get("before_path"):
            info = encode_image(pair["before_path"], megapixels)
            content.append({
                "type": "text",
                "text": f"对比图组{idx} ({name}) — "
                        "AI上一轮标注结果(红框，含错误):"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": info["url"]}
            })

        if pair.get("after_path"):
            info = encode_image(pair["after_path"], megapixels)
            content.append({
                "type": "text",
                "text": f"对比图组{idx} ({name}) — "
                        "用户人工修正后的正确答案(绿框):"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": info["url"]}
            })

    messages = [
        {"role": "system", "content": PROMPT_GEN_SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]

    response = call_qwen(
        client, model, messages, max_retry, log, stop_event
    )

    data = json.loads(response.choices[0].message.content)

    prompt = str(data.get("prompt", "")).strip()

    if not prompt:
        raise RuntimeError("AI未返回有效提示词")

    # 负样本排除(AI未返回或返回空则视为None，调用方保留原值)
    negative = str(data.get("negative") or "").strip() or None

    return prompt, negative


# 查询账号当前可用的模型列表(按名称排序)；失败抛异常
def fetch_available_models(api_key=None, base_url=BASE_URL):

    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError("未设置 API Key")

    client = OpenAI(api_key=api_key, base_url=base_url or BASE_URL)

    models = client.models.list()

    ids = [m.id for m in models.data]

    return sorted(ids)


# ============================================================
# 类别表解析
#
# 提示词内置类别表格式(每个类别一行)，支持三种分隔符:
#   标签名：描述 / 标签名:描述 / 标签名=描述
# 标签名的声明顺序即YOLO类别id顺序
# ============================================================

# 常见提示词元词，不算标签名
META_WORDS = {
    "要求", "说明", "注意", "规则", "格式", "提示",
    "示例", "任务", "类别", "目标", "描述", "其他",
    "补充", "输出", "参数", "约束", "背景", "场景",
    "负样本排除",
}


def parse_category_list(prompt):
    """解析提示词中的类别表，返回[(标签名, 描述), ...]

    标签名原样保留——冒号前的名称就是classes.txt里存的
    标签名(纯数字编号也是合法标签名，不得改名)。
    """

    cats = []

    for raw in prompt.splitlines():

        line = raw.strip()

        # 支持三种分隔符: 中文冒号/等号/英文冒号
        for sep in ("：", "=", ":"):
            if sep in line:
                name, desc = line.split(sep, 1)
                break
        else:
            continue

        name = name.strip().rstrip("|\\ \t").strip()
        desc = desc.strip().rstrip("|\\ \t，,；;").strip()

        if not name or not desc or len(desc) < 2:
            continue

        if len(name) > 30:
            continue

        if re.search(r"[。；，,！!？?\"]", name):
            continue

        if name in META_WORDS:
            continue

        if any(
            name.startswith(w) and len(name) > len(w)
            for w in META_WORDS
        ):
            continue

        if any(
            k in name for k in
            ("不要", "排除", "禁止", "忽略", "以下", "如下")
        ):
            continue

        if any(name == n for n, _ in cats):
            continue

        cats.append((name, desc))

    return cats


def build_category_lookup(cats):
    """构建 标签名->声明序号 的匹配表"""
    lookup = {}
    names = []

    for name, _ in cats:
        if name not in lookup:
            lookup[name] = len(names)
            names.append(name)

    return lookup, names


# 严格类别匹配: AI输出的类名 -> 类别表中声明的标签名。
# 依次尝试: 逐字一致 -> 忽略大小写/首尾空白 ->
# 唯一的包含关系匹配。匹配不上返回None，调用方丢弃该框。
def resolve_category(cls, cat_lookup):

    if cls in cat_lookup:
        return cls

    key = cls.strip().casefold()

    for name in cat_lookup:
        if name.strip().casefold() == key:
            return name

    hits = [
        name for name in cat_lookup
        if name in cls or cls in name
    ]

    if len(hits) == 1:
        return hits[0]

    return None


# ============================================================
# 手动标注 -> 示例图导出
#
# 用户在"标注"页面手动标注若干图片后，把DB中的框和类别名
# 画到图片副本上，作为few-shot参考示例喂给"AI生成提示词"。
# ============================================================

def _load_example_font():
    for font_path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, 28)
        except Exception:
            continue
    return ImageFont.load_default()


def export_manual_examples(project_id, log=print):
    """把项目中已标注的图片导出为带框示例图

    每张有标注的图片生成一份副本(框+类别名画在图上)，
    存到 项目目录/ai_examples/ (文件名ai_开头，
    不会被启动同步清理)。返回示例图路径列表。
    """
    out_dir = None
    results = []

    font = _load_example_font()

    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT i.id, i.filename, i.storage_path FROM images i "
            "JOIN annotations a ON a.image_id = i.id "
            "WHERE i.project_id = ? GROUP BY i.id",
            (project_id,)
        )
        image_rows = cursor.fetchall()

        for img_id, filename, storage_path in image_rows:
            cursor.execute(
                "SELECT type, data, class_name, attributes FROM annotations "
                "WHERE image_id = ?",
                (img_id,)
            )
            ann_rows = cursor.fetchall()

            # 只导出含手动bbox标注的图片(AI生成的框不作为示例，
            # 避免把AI自己的结果喂回去)
            bboxes = []
            for ann_type, data_json, class_name, attrs_json in ann_rows:
                if ann_type != 'bbox':
                    continue
                try:
                    attrs = json.loads(attrs_json or '{}')
                except (ValueError, TypeError):
                    attrs = {}
                if attrs.get('source') == 'ai':
                    continue
                try:
                    d = json.loads(data_json)
                    x = float(d.get('x', 0))
                    y = float(d.get('y', 0))
                    w = float(d.get('width', 0))
                    h = float(d.get('height', 0))
                except (ValueError, TypeError):
                    continue
                if w > 1 and h > 1:
                    bboxes.append((class_name, x, y, w, h))

            if not bboxes:
                continue

            try:
                img = Image.open(storage_path).convert("RGB")
            except Exception as e:
                log(f"[警告] 示例图读取失败({filename}): {e}")
                continue

            if out_dir is None:
                out_dir = Path(storage_path).parent / "ai_examples"
                out_dir.mkdir(exist_ok=True)

            draw = ImageDraw.Draw(img)
            # 线宽随图幅缩放
            lw = max(3, min(img.size) // 300)

            # 同类同色
            class_colors = {}

            for class_name, x, y, w, h in bboxes:
                if class_name not in class_colors:
                    rng = random.Random(class_name)
                    class_colors[class_name] = (
                        rng.randint(50, 220),
                        rng.randint(50, 220),
                        rng.randint(50, 220),
                    )
                color = class_colors[class_name]

                x2, y2 = x + w, y + h
                draw.rectangle([x, y, x2, y2], outline=color, width=lw)

                if class_name:
                    ty = y - 34
                    if ty < 0:
                        ty = y + 2
                    draw.text((x, ty), class_name, fill=color, font=font)

            stem = Path(filename).stem
            out_path = out_dir / f"ai_example_{stem}.jpg"
            img.save(out_path)
            results.append(str(out_path))

    return results


def export_locked_feedback(project_id, only_ids=None, log=print):
    """导出已人工锁定(确认)图片的全部标注为带框反馈参考图

    锁定图 = 用户确认过的正确标注(包括修正过的AI框)。
    用于"AI大规模标错 -> 拿确认过的图重新喂给AI更新提示词"。
    存到 项目目录/ai_examples/ (文件名ai_fb_开头，不会被
    启动同步清理)。返回(示例图路径列表, 图片数)。
    """
    out_dir = None
    results = []

    font = _load_example_font()

    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, filename, storage_path FROM images "
            "WHERE project_id = ? AND locked = 1 ORDER BY id",
            (project_id,)
        )
        image_rows = [
            (r[0], r[1], r[2]) for r in cursor.fetchall()
            if only_ids is None or r[0] in only_ids
        ]

        for img_id, filename, storage_path in image_rows:
            cursor.execute(
                "SELECT type, data, class_name FROM annotations "
                "WHERE image_id = ?",
                (img_id,)
            )
            ann_rows = cursor.fetchall()

            # 锁定图上的全部bbox都是确认过的正确答案，
            # 不按source过滤(修正过的AI框也算)
            bboxes = []
            for ann_type, data_json, class_name in ann_rows:
                if ann_type != 'bbox':
                    continue
                try:
                    d = json.loads(data_json)
                    x = float(d.get('x', 0))
                    y = float(d.get('y', 0))
                    w = float(d.get('width', 0))
                    h = float(d.get('height', 0))
                except (ValueError, TypeError):
                    continue
                if w > 1 and h > 1:
                    bboxes.append((class_name, x, y, w, h))

            if not bboxes:
                continue

            try:
                img = Image.open(storage_path).convert("RGB")
            except Exception as e:
                log(f"[警告] 反馈图读取失败({filename}): {e}")
                continue

            if out_dir is None:
                out_dir = Path(storage_path).parent / "ai_examples"
                out_dir.mkdir(exist_ok=True)

            draw = ImageDraw.Draw(img)
            lw = max(3, min(img.size) // 300)

            class_colors = {}
            for class_name, x, y, w, h in bboxes:
                if class_name not in class_colors:
                    rng = random.Random(class_name)
                    class_colors[class_name] = (
                        rng.randint(50, 220),
                        rng.randint(50, 220),
                        rng.randint(50, 220),
                    )
                color = class_colors[class_name]

                x2, y2 = x + w, y + h
                draw.rectangle([x, y, x2, y2], outline=color, width=lw)

                if class_name:
                    ty = y - 34
                    if ty < 0:
                        ty = y + 2
                    draw.text((x, ty), class_name, fill=color, font=font)

            stem = Path(filename).stem
            out_path = out_dir / f"ai_fb_{stem}.jpg"
            img.save(out_path)
            results.append(str(out_path))

    return results, len(results)


# 把bbox列表画到图片副本上(用于修正对比查看/导出)
# bboxes: [(class_name, x1, y1, x2, y2), ...]
# color: None=同类同色随机，否则所有框用该固定颜色
def _render_boxes(storage_path, bboxes, color=None):

    img = Image.open(storage_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    lw = max(3, min(img.size) // 300)

    class_colors = {}
    for class_name, x1, y1, x2, y2 in bboxes:

        if color is not None:
            c = color
        else:
            if class_name not in class_colors:
                rng = random.Random(class_name)
                class_colors[class_name] = (
                    rng.randint(50, 220),
                    rng.randint(50, 220),
                    rng.randint(50, 220),
                )
            c = class_colors[class_name]

        draw.rectangle([x1, y1, x2, y2], outline=c, width=lw)

        if class_name:
            ty = y1 - 34
            if ty < 0:
                ty = y1 + 2
            draw.text((x1, ty), class_name, fill=c, font=_load_example_font())

    return img


# 读取图片AI原始框快照 -> [(class_name, x1, y1, x2, y2)]
def _get_ai_snapshot_bboxes(img_id):

    snapshot = db.get_image_ai_original(img_id)
    bboxes = []

    if isinstance(snapshot, list):
        for obj in snapshot:
            try:
                bb = [float(v) for v in obj["bbox"]]
                if bb[2] - bb[0] > 1 and bb[3] - bb[1] > 1:
                    bboxes.append((obj.get("class_name", ""), *bb))
            except (KeyError, TypeError, ValueError):
                continue

    return bboxes


# 读取图片当前全部bbox标注(不按source过滤)
# -> [(class_name, x1, y1, x2, y2)]
def _get_current_bboxes(img_id):

    bboxes = []

    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data, class_name FROM annotations "
            "WHERE image_id = ?",
            (img_id,)
        )
        for data_json, class_name in cursor.fetchall():
            try:
                d = json.loads(data_json)
                x = float(d.get('x', 0))
                y = float(d.get('y', 0))
                w = float(d.get('width', 0))
                h = float(d.get('height', 0))
            except (ValueError, TypeError):
                continue
            if w > 1 and h > 1:
                bboxes.append((class_name, x, y, x + w, y + h))

    return bboxes


def export_modified_comparison(project_id, only_ids=None, log=print):
    """导出"AI原始结果 vs 用户修正结果"对比图组

    针对 modified=1 且有AI快照(ai_original)的图片，每张生成两张图:
      ai_diff_{stem}_before.jpg — AI上一轮原始结果(红框)
      ai_diff_{stem}_after.jpg  — 用户人工修正后的当前标注(绿框)
    存到 项目目录/ai_examples/ (ai_前缀不会被启动同步清理)。
    返回 [{"image_id", "filename", "before_path", "after_path"}, ...]
    """
    RED = (220, 40, 40)
    GREEN = (40, 180, 60)
    pairs = []

    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, filename, storage_path FROM images "
            "WHERE project_id = ? AND modified = 1 "
            "AND ai_original IS NOT NULL ORDER BY id",
            (project_id,)
        )
        image_rows = [
            (r[0], r[1], r[2]) for r in cursor.fetchall()
            if only_ids is None or r[0] in only_ids
        ]

    for img_id, filename, storage_path in image_rows:

        before = _get_ai_snapshot_bboxes(img_id)

        # 修正后 = 当前DB标注(全部bbox，不按source过滤)
        after = _get_current_bboxes(img_id)

        if not before and not after:
            continue

        if not Path(storage_path).is_file():
            log(f"[警告] 对比图读取失败，文件不存在({filename})")
            continue

        try:
            out_dir = Path(storage_path).parent / "ai_examples"
            out_dir.mkdir(exist_ok=True)
            stem = Path(filename).stem

            before_path = None
            after_path = None
            if before:
                before_path = out_dir / f"ai_diff_{stem}_before.jpg"
                _render_boxes(storage_path, before, RED).save(before_path)
            if after:
                after_path = out_dir / f"ai_diff_{stem}_after.jpg"
                _render_boxes(storage_path, after, GREEN).save(after_path)
        except Exception as e:
            log(f"[警告] 对比图生成失败({filename}): {e}")
            continue

        pairs.append({
            "image_id": img_id,
            "filename": filename,
            "before_path": str(before_path) if before_path else None,
            "after_path": str(after_path) if after_path else None,
        })

    return pairs


# ============================================================
# 数据库集成: 项目类别注册
# ============================================================

def _random_class_color():
    return f"#{random.randint(40, 220):02x}{random.randint(40, 220):02x}{random.randint(40, 220):02x}"


def register_classes(project_id, class_names):
    """把类别名注册进项目classes表，返回 名称->项目类别id 映射。

    已存在的类别保持原id不变；新类别追加到末尾。
    """
    project = db.get_project(project_id)

    if not project:
        raise RuntimeError(f"项目不存在: {project_id}")

    classes = project.get('classes') or '[]'

    if isinstance(classes, str):
        try:
            classes = json.loads(classes)
        except Exception:
            classes = []

    changed = False

    for name in class_names:
        if not any(c.get('name') == name for c in classes):
            cid = max([c.get('id', 0) for c in classes], default=-1) + 1
            classes.append({
                'id': cid,
                'name': name,
                'color': _random_class_color()
            })
            changed = True

    if changed:
        db.update_project(project_id, classes=classes)

    return {c['name']: c['id'] for c in classes}


# ============================================================
# 核心标注流程（数据库版）
#
# params:
#   project_id        项目id
#   images            项目图片记录列表(db.get_project_images返回的dict)
#   prompt / negative / example_images / model
#   api_key / base_url
#   detect_rounds / max_workers / upload_megapixels / max_retry
#   zoom_refine / zoom_iters
#   limit             试标数量(>0时只标注前N张未完成的)
#   overwrite         是否覆盖已有标注
#   project_dir       项目存储目录(存断点)
# log: 日志回调 log(str)
# stop_event: threading.Event
# progress: 进度回调 progress(done, total)
# ============================================================

def run_ai_annotation(params, log, stop_event=None, progress=None):

    project_id = params["project_id"]
    all_records = params["images"]
    prompt = params.get("prompt", "")
    model = params.get("model", DEFAULT_MODEL)
    api_key = params.get("api_key", "")
    base_url = params.get("base_url", BASE_URL)
    detect_rounds = max(1, params.get("detect_rounds", DEFAULT_DETECT_ROUNDS))
    max_workers = params.get("max_workers", DEFAULT_MAX_WORKERS)
    megapixels = params.get("upload_megapixels", DEFAULT_UPLOAD_MEGAPIXELS)
    max_retry = params.get("max_retry", DEFAULT_MAX_RETRY)
    zoom_refine = params.get("zoom_refine", DEFAULT_ZOOM_REFINE)
    zoom_iters = max(1, params.get("zoom_iters", DEFAULT_ZOOM_ITERS))
    limit = max(0, params.get("limit", 0))
    overwrite = params.get("overwrite", False)
    project_dir = Path(params.get("project_dir", "."))

    negative = (params.get("negative", "") or "").strip()

    # ===== 示例图片（few-shot）=====
    example_images = params.get("example_images") or []

    example_paths = []

    for p in example_images:
        path = Path(p)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            example_paths.append(path)
        else:
            log(f"[警告] 示例图片无效或格式不支持，已跳过: {p}")

    if not prompt and not example_paths:
        raise RuntimeError("请填写标注任务描述或提供示例图片")

    if not prompt:
        prompt = (
            "Annotate all objects of the categories shown in the "
            "reference example images, following the same "
            "annotation style."
        )

    # ===== 解析提示词中的类别表 =====
    cats = parse_category_list(prompt)

    cat_lookup, _ = build_category_lookup(cats)

    if negative:
        prompt = (
            prompt.rstrip()
            + "\n负样本排除(下列东西即使出现也不要框): "
            + negative
        )
        log(f"负样本排除: {negative}")

    if cats:
        prompt = (
            prompt.rstrip()
            + "\n\n类别输出约束(最高优先级):\n"
            "只检测上面类别表中列出的类别。结果中的"
            "label字段必须逐字使用类别表中的标签名"
            "(冒号前的名称)，不要翻译、改名、合并类别。"
            "类别表中没有的类别绝对不允许输出: 严禁自创、"
            "臆造任何类别名，也严禁使用类别名的中文直译或"
            "其他变体；不属于任何已知类别的物体直接忽略，"
            "宁可少输出也不能错标类别。"
        )
        log(f"类别表: {len(cats)}个类别 -> "
            + ", ".join(name for name, _ in cats))

    # 发给模型的完整任务文本(每张图的请求都会携带)
    task_text = "任务:\n" + prompt

    if example_paths:
        if cats:
            task_text += (
                "\n\n标记为\"Reference example\"的图片是标注示例:"
                "它只用于演示框要贴多紧、场景中哪部分算作目标。"
                "必须完全照抄示例的框的贴紧程度: 你的框必须像"
                "示例一样紧贴目标轮廓，绝不能更大。"
                "但label不要照抄示例图上画的手写标签文字:"
                "label必须使用任务类别表中冒号前的标签名。"
            )
        else:
            task_text += (
                "\n\n标记为\"Reference example\"的图片是标注示例:"
                "其中的框和标签定义了目标类别、场景中哪部分算作"
                "目标、以及框要贴多紧。必须完全照抄示例的框的"
                "贴紧程度，并逐字复制示例标签上的类别名，"
                "无论它是什么语言。"
            )

    # 保存最终生效的提示词，便于复查
    project_dir.mkdir(parents=True, exist_ok=True)
    with open(project_dir / "ai_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt + "\n")

    if not cats:
        log("[提示] 未在提示词中解析到类别表(每行\"标签名=描述\""
            "或\"标签名：描述\")，严格类别校验已关闭，"
            "AI可能自创类别")

    if stop_event is None:
        stop_event = threading.Event()

    client = make_client(api_key, base_url)

    # 预编码示例图(只编码一次，避免每张图的请求重复编码)
    example_parts = []

    for ex_path in example_paths:
        ex_info = encode_image(ex_path, megapixels)
        example_parts.append({
            "type": "text",
            "text": "Reference example (标注示例，不要在此图上找目标)"
        })
        example_parts.append({
            "type": "image_url",
            "image_url": {"url": ex_info["url"]}
        })

    # ===== 断点文件 =====

    progress_file = project_dir / "ai_progress.json"

    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            progress_data = json.load(f)
    else:
        progress_data = {"completed": []}

    completed = set(progress_data["completed"])

    # ===== 图片列表 =====

    records = [
        r for r in all_records
        if Path(r.get('storage_path') or '').suffix.lower() in IMAGE_EXTENSIONS
    ]

    # 人工锁定(手动修改过)的图片整图跳过,不交给AI
    locked_records = [r for r in records if r.get('locked')]
    if locked_records:
        log(f"[跳过] {len(locked_records)} 张图片已人工锁定,不参与AI标注")
    records = [r for r in records if not r.get('locked')]

    total_all = len(records)

    records = [
        r for r in records
        if r['filename'] not in completed
    ]

    # 试标模式: 只处理前limit张(已完成的不占名额)
    if limit > 0:
        records = records[:limit]

    log("=" * 60)
    log(f"总图片数量: {total_all}")
    log(f"待处理数量: {len(records)}")
    if limit > 0:
        log(f"试标模式: 本次仅标注前{limit}张(不含已完成)，"
            f"实际待处理{len(records)}张")
    if cats:
        log(f"类别表已启用: 类别id按声明顺序注册到项目")
    if example_paths:
        log(f"示例图片(few-shot): {len(example_paths)} 张")
        for p in example_paths:
            log(f"  - {p.name}")
    if detect_rounds > 1:
        log(f"检测次数: {detect_rounds} (结果取平均)")
    log(f"二次放大精修: {'开启' if zoom_refine else '关闭'}"
        + (f" ({zoom_iters}轮迭代)" if zoom_refine else ""))
    log(f"覆盖已有标注: {'是' if overwrite else '否'}")
    log(f"模型: {model}  并发: {max_workers}")
    log("=" * 60)

    if len(records) == 0:
        log("所有图片已经完成")
        return

    done = 0
    if progress is not None:
        progress(done, len(records))

    # ===== 并发处理 =====
    # 每张图独立单图请求；检测与二次放大精修都在
    # 工作线程内完成并相互并行

    def process_image(image_record):

        image_path = Path(image_record['storage_path'])

        img_start = time.time()

        log(f"正在请求Qwen: {image_path.name}")

        # 多轮检测: 同一张图重复请求，结果取平均抵消抖动
        round_objs = []
        info = None

        for round_idx in range(detect_rounds):

            if round_idx > 0 and stop_event.is_set():
                raise StopRequested()

            try:
                objs, info = detect_single(
                    client, model, image_path, task_text,
                    example_parts, megapixels, max_retry,
                    log, stop_event
                )
                round_objs.append(objs)
            except StopRequested:
                raise
            except Exception as e:
                # 单轮失败容忍，其余轮结果可用即可
                log(
                    f"{image_path.name} "
                    f"第{round_idx+1}轮检测失败: {e}"
                )

        if not round_objs:
            raise RuntimeError(
                f"{image_path.name} 所有检测轮次均失败"
            )

        objects = merge_round_objects(round_objs, len(round_objs))

        # ===== 二次放大精修 =====
        if zoom_refine and objects:
            try:
                zoom_refine_objects(
                    client, model, image_path,
                    objects, max_retry,
                    log, stop_event, iters=zoom_iters,
                    task_text=task_text
                )
            except StopRequested:
                raise
            except Exception as e:
                log(
                    f"[警告] 二次精修失败({image_path.name}): "
                    f"{e}，使用一次检测结果"
                )

        elapsed = time.time() - img_start

        return {
            "record": image_record,
            "objects": objects,
            "info": info,
            "elapsed": elapsed,
        }

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    )

    futures = [
        executor.submit(process_image, record)
        for record in records
    ]

    stopped = False

    # 已告警过的非法类名(避免日志刷屏)
    warned_labels = set()

    try:
        for future in concurrent.futures.as_completed(futures):

            if stop_event.is_set():
                stopped = True
                break

            try:
                item = future.result()
            except StopRequested:
                stopped = True
                break
            except Exception as e:
                log(f"图片处理失败，跳过: {e}")
                continue

            record = item["record"]
            objects = item["objects"]
            image_path = Path(record['storage_path'])

            if not image_path.exists():
                continue

            # ===== 严格类别校验 =====
            # 提示词类别表之外的类名一律丢弃
            if cats:
                valid_objects = []

                for obj in objects:
                    cls = obj["class_name"]

                    resolved = resolve_category(cls, cat_lookup)

                    if resolved is None:
                        if cls not in warned_labels:
                            warned_labels.add(cls)
                            log(
                                f"[丢弃] AI输出的类名 \"{cls}\" "
                                f"不在类别表中，该框已丢弃"
                                f"(同类名后续不再重复提示)"
                            )
                        continue

                    # 统一回写为类别表中的标准标签名
                    obj["class_name"] = resolved
                    valid_objects.append(obj)

                objects = valid_objects

            # ===== 注册类别到项目 =====
            class_names = sorted({o["class_name"] for o in objects})
            name_to_id = register_classes(project_id, class_names)

            # ===== 写入数据库 =====
            if overwrite:
                db.delete_image_annotations(record['id'])
            else:
                # 只清除上一轮AI生成的标注，避免重标产生重复框；
                # 手动标注不受影响
                removed = db.delete_ai_annotations(record['id'])
                if removed:
                    log(f"    已清除上一轮AI标注 {removed} 条")

            for obj in objects:
                x1, y1, x2, y2 = obj["bbox"]
                db.add_annotation(
                    record['id'],
                    project_id,
                    name_to_id.get(obj["class_name"], 0),
                    obj["class_name"],
                    'bbox',
                    {
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1,
                    },
                    {'source': 'ai', 'confidence': obj.get('confidence', 1.0)}
                )

            # 保存AI原始框快照(用户之后人工修正时，
            # 用于生成"AI原始结果 vs 修正结果"对比图反馈)
            try:
                db.set_image_ai_original(record['id'], [
                    {
                        "class_name": o["class_name"],
                        "bbox": [float(v) for v in o["bbox"]],
                    }
                    for o in objects
                ])
            except Exception as e:
                log(f"[警告] AI快照保存失败({record['filename']}): {e}")

            # ===== 更新断点 =====
            completed.add(record['filename'])

            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"completed": list(completed)},
                    f, indent=4
                )

            done += 1
            if progress is not None:
                progress(done, len(records))

            log(
                f"[{done}/{len(records)}] "
                f"{record['filename']}: -> {len(objects)} 框 "
                f"({item['elapsed']:.2f}s)"
            )

    finally:
        # 取消尚未开始的任务
        executor.shutdown(wait=False, cancel_futures=True)

    if stopped:
        log("已停止（已完成部分保留断点，可继续续标）")
        return

    log("")
    log("=" * 60)
    if limit > 0:
        log(f"试标批次完成({len(records)}张)")
        log("请切换到\"标注\"页面查看AI标注效果(可直接拖动调整框):")
        log("  满意 -> 将\"试标数量\"改回0，点开始标注继续剩余图片(断点自动续标)")
        log("  不满意 -> 修改任务描述/示例图片后，点击\"清除断点\"重新试标")
    else:
        log("全部完成")
    log(f"本次处理: {len(records)} 张")
    log("=" * 60)


# 清除断点记录(重新标注)
def clear_breakpoint(project_dir):
    progress_file = Path(project_dir) / "ai_progress.json"
    if progress_file.exists():
        progress_file.unlink()
        return True
    return False
