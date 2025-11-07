# -*- coding: utf-8 -*-
import os

import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel, AutoTokenizer
from transformers import CLIPProcessor, CLIPModel
import torch
from Pytorch_Retinaface.detect import init_detector, detect_face
from utils import process_img_beauty, calculate_face_metrics
from PIL import Image, ImageDraw, ImageFont


def human_beauty_infer(pixel_values, tokenizer, model, output_level=1, data_type=torch.float16):
    """
        输出 4 类结果：
        1. 快速美学评分（整体颜值分）；
        2. 元投票评分（更精确的综合分，耗时更长）；
        3. 12 个维度的专家评分（如面部结构、肤色、妆容等）；
        4. 12 个维度的专家文字注释（对各维度的具体描述）。
    """
    pixel_values = pixel_values.to(dtype=data_type, device=model.device)
    # 在这里修改美学大模型参数
    generation_config = dict(max_new_tokens=256,
                             do_sample=False,
                             temperature=0.0)
    question = '<image>\nRate the aesthetics of this human picture.'

    # infer
    output = {}
    if output_level >= 1:
        pred_score = model.score(tokenizer, pixel_values, question)
        output['整体颜值评分'] = pred_score
    if output_level >= 2:
        metavoter_score = model.run_metavoter(tokenizer, pixel_values)  # 精确评分
        output['精确综合评分'] = metavoter_score
    if output_level >= 3:
        expert_score, expert_text = model.expert_score(tokenizer, pixel_values)  # 12个维度的专家评分，专家输出文本
        output['12个维度专家评分'] = expert_score
    if output_level >= 4:
        expert_annotataion = model.expert_annotataion(tokenizer, pixel_values, generation_config)  # 各维度具体描述
        output['12个维度具体描述'] = expert_annotataion
    return output


class Models:
    def __init__(self, select_model2path={}, select_model2column={}, clip_prompts={}, output_leval=1, no_half=False):
        self.model2path = select_model2path        # 模型名：路径
        self.model2column = select_model2column    # 模型名：列名
        self.processor_dict = {}
        self.model_dict = {}
        self.clip_prompts = clip_prompts    # clip模型的提示模板
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_level = output_leval
        if no_half:
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16

    def load_models(self):
        """加载所有模型"""
        for model_name, model_path in self.model2path.items():
            if model_name == "human_beauty":
                # 人脸美学大模型
                self.processor_dict[model_name] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, local_files_only=True)
                self.model_dict[model_name] = AutoModel.from_pretrained(
                                                model_path,
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                                use_flash_attn=False,
                                                trust_remote_code=True,
                                                local_files_only=True).eval().to(self.device)
                print(f"当前推理设备：{self.model_dict[model_name].device}（若为'cpu'，1B模型会极慢，需要用GPU）")
            elif model_name == "clip_vit":
                # clip模型
                self.processor_dict[model_name] = CLIPProcessor.from_pretrained(model_path)
                self.model_dict[model_name] = CLIPModel.from_pretrained(model_path).to(dtype=self.data_type, device=self.device).eval()
            else:
                # 其他模型
                self.processor_dict[model_name] = AutoImageProcessor.from_pretrained(model_path)
                self.model_dict[model_name] = AutoModelForImageClassification.from_pretrained(model_path).to(dtype=self.data_type, device=self.device).eval()

    def infer_imgs(self, batch_imgs):
        """推理图片"""
        batch_size = len(batch_imgs)
        batch_results = [{} for _ in range(batch_size)]
        # 每个模型分别推理一次图片
        for model_name in self.model_dict.keys():
            processor = self.processor_dict[model_name]
            model = self.model_dict[model_name]
            if model_name == "human_beauty":
                # 美学大模型
                max_num = 4     # 子图数量，子图越多推理越慢
                pixel_values_list = [process_img_beauty(img, max_num=max_num) for img in batch_imgs]
                pixel_values = torch.cat(pixel_values_list, dim=0).to(dtype=self.data_type, device=self.device)
                with torch.no_grad():
                    result = human_beauty_infer(pixel_values, processor, model, self.output_level, self.data_type)
                for i in range(batch_size):
                    batch_results[i].update(result)
            elif model_name == "clip_vit":
                # clip大模型
                for column, prompt in self.clip_prompts.items():
                    inputs = processor(text=prompt, images=batch_imgs, return_tensors="pt", padding=True)
                    inputs = {k: v.to(dtype=self.data_type, device=self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image  # 图文相似度
                    probs = logits_per_image.softmax(dim=1)  # 标签概率
                    index = torch.argmax(probs, dim=1).item().long
                    for i in range(batch_size):
                        batch_results[i][column] = index[i].item()  # 结果保留为prompts中对应的类别下标
            else:
                # 其他
                inputs = processor(images=batch_imgs, return_tensors="pt")
                inputs = {k: v.to(dtype=self.data_type, device=self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    output = model(**inputs)
                logits = output.logits
                predicted_class_idx = logits.argmax(-1).tolist()  # 预测的类别id
                # 保留两个结果：类别id和类别名
                column = self.model2column[model_name]
                for i in range(batch_size):
                    batch_results[i][column] = predicted_class_idx[i]  # 列名：类别id
                    batch_results[i][model_name] = model.config.id2label[predicted_class_idx[i]]  # 模型名：类别名

        return batch_results


def load_Retinaface():
    params = {
        # 模型相关（固定使用resnet50）
        # "trained_model": "./Pytorch_Retinaface/weights/Resnet50_Final.pth",  # 模型路径
        # "network": "resnet50",  # 固定为resnet50
        "trained_model": "./Pytorch_Retinaface/weights/mobilenet0.25_Final.pth",
        "network": "mobile0.25",
        "cpu": False,  # 是否用CPU（False表示用GPU，True表示用CPU）
        # 检测阈值参数
        "confidence_threshold": 0.02,
        "top_k": 1000,
        "nms_threshold": 0.4,
        "keep_top_k": 100,
        "vis_thres": 0.1,  # 可视化/保留结果的置信度阈值，设小点，后面找最高的处理
    }

    # 为了兼容原detect.py的参数格式，将字典转为类对象
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Args(**params)
    net, cfg, device = init_detector(args)
    return net, cfg, device, args


def Retinaface_infer(img_path, img_output_dir, net, cfg, device, args, save_img):
    img = Image.open(img_path).convert("RGB")

    # 限制图像尺寸（长边不超过800像素）
    max_size = 256  # 可根据显存情况调整
    width, height = img.size
    # 计算缩放比例（保持宽高比）
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    img_np = np.array(img)[:, :, ::-1].astype(np.float32)  # RGB→BGR
    dets = detect_face(net, cfg, device, img_np, args)
    output = {}
    if dets is not None and len(dets) > 0:
        # 找到置信度最高的检测结果，按置信度降序排序，取第一个
        max_conf_det = max(dets, key=lambda x: x[4])
        landmarks = max_conf_det[5:15]             # 提取关键点（det[5:15]为5个关键点的坐标）
        # 计算脸型和比例
        res = calculate_face_metrics(landmarks)
        output = {
            '脸型': res['face_shape'],
            '宽高比': res['width_height_ratio'],
            '五眼比例': res['five_eye_ratio'],
            '三庭比例': res['three_court_ratio']
        }

        if save_img:
            # 可视化标注
            draw = ImageDraw.Draw(img)
            x1, y1, x2, y2 = map(int, max_conf_det[:4])
            draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)  # 红色（RGB）
            # 画关键点
            for j in range(5):
                x, y = int(landmarks[2 * j]), int(landmarks[2 * j + 1])
                draw.ellipse([(x-3, y-3), (x+3, y+3)], fill=(0, 255, 0))  # 绿色（RGB）

            img_filename = os.path.basename(img_path)
            img_output_path = os.path.join(img_output_dir, img_filename)
            img.save(img_output_path)
            # print(f"脸型结果图像已保存至：{output_path}")

    return output

