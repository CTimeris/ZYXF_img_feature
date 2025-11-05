# -*- coding: utf-8 -*-
import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel, AutoTokenizer
from transformers import CLIPProcessor, CLIPModel
import torch
from Pytorch_Retinaface.detect import init_detector, detect_face
from utils import process_img_beauty, calculate_face_metrics
from PIL import Image


def human_beauty_infer(pixel_values, tokenizer, model):
    """
        输出 4 类结果：
        1. 快速美学评分（整体颜值分）；
        2. 元投票评分（更精确的综合分，耗时更长）；
        3. 12 个维度的专家评分（如面部结构、肤色、妆容等）；
        4. 12 个维度的专家文字注释（对各维度的具体描述）。
    """
    pixel_values = pixel_values.to(dtype=torch.float16, device=model.device)
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    question = '<image>\nRate the aesthetics of this human picture.'

    # infer
    pred_score = model.score(tokenizer, pixel_values, question)
    metavoter_score = model.run_metavoter(tokenizer, pixel_values)  # 精确评分
    expert_score, expert_text = model.expert_score(tokenizer, pixel_values)  # 12个维度的专家评分，专家输出文本
    expert_annotataion = model.expert_annotataion(tokenizer, pixel_values, generation_config)  # 各维度具体描述
    output = {
        '整体颜值评分': pred_score,
        '精确综合评分': metavoter_score,
        '12个维度专家评分': expert_score,
        # '专家文本': expert_text,
        '12个维度具体描述': expert_annotataion
    }
    return output


def clip_infer(image, text, processor, model):
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 图文相似度
    probs = logits_per_image.softmax(dim=1)  # 标签概率
    return probs


class Models:
    def __init__(self, args):
        self.args = args
        self.model2path = args.select_model2path        # 模型名：路径
        self.model2column = args.select_model2column    # 模型名：列名
        self.processor_dict = {}
        self.model_dict = {}
        self.clip_prompts = args.clip_prompts    # clip模型的提示模板

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
                                                local_files_only=True).eval().cuda()
                print(f"当前推理设备：{self.model_dict[model_name].device}（若为'cpu'，1B模型会极慢，需要用GPU）")
            elif model_name == "clip_vit":
                # clip模型
                self.processor_dict[model_name] = CLIPProcessor.from_pretrained(model_path)
                self.model_dict[model_name] = CLIPModel.from_pretrained(model_path)
            elif model_name == "Retinaface":
                net, cfg, Retinaface_device, Retinaface_args = load_Retinaface()  # 加载Retinaface模型
                Retinaface_model = [net, cfg, Retinaface_device, Retinaface_args]   # 把模型所有配置放到一个字典
                self.processor_dict[model_name] = None
                self.model_dict[model_name] = Retinaface_model
            else:
                # 其他模型
                self.processor_dict[model_name] = AutoImageProcessor.from_pretrained(model_path)
                self.model_dict[model_name] = AutoModelForImageClassification.from_pretrained(model_path)
        print("所有模型加载完毕")

    def infer_one_img(self, img_path, img_output_path):
        """推理一张图片"""
        if not self.model_dict or not self.processor_dict:
            print("先使用.load_model()加载模型")

        img = Image.open(img_path).convert("RGB")

        results = {}
        # 每个模型分别推理一次图片
        for model_name in self.model_dict.keys():
            processor = self.processor_dict[model_name]
            model = self.model_dict[model_name]
            if model_name == "human_beauty":
                # 美学大模型
                pixel_values = process_img_beauty(img)
                result = human_beauty_infer(pixel_values, processor, model)
                results.update(result)
            elif model_name == "clip_vit":
                # clip大模型
                for column, prompt in self.clip_prompts.items():
                    probs = clip_infer(img, prompt, processor, model)
                    probs_flat = probs.squeeze()
                    index = torch.argmax(probs_flat).item()
                    results[column] = index     # 结果保留为prompts中对应的类别下标
            elif model_name == "Retinaface":
                # Retinaface
                [net, cfg, Retinaface_device, Retinaface_args] = self.model_dict[model_name]
                save_img = self.args.save_img
                result = Retinaface_infer(img_path, img_output_path, net, cfg, Retinaface_device, Retinaface_args, save_img)
                results.update(result)
            else:
                # 其他
                input_data = processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    output = model(**input_data)
                    logits = output.logits
                    predicted_class_idx = logits.argmax(-1).item()                  # 预测的类别id
                    predicted_class = model.config.id2label[predicted_class_idx]    # id对应的类别名
                    # 保留两个结果：类别id和类别名
                    column = self.model2column[model_name]
                    results[column] = predicted_class_idx           # 列名：类别id
                    results[model_name] = predicted_class           # 模型名：类别名
        return results


def load_Retinaface():
    params = {
        # 模型相关（固定使用resnet50）
        "trained_model": "./Pytorch_Retinaface/weights/Resnet50_Final.pth",  # 模型路径
        "network": "resnet50",  # 固定为resnet50
        "cpu": False,  # 是否用CPU（False表示用GPU，True表示用CPU）
        # 检测阈值参数
        "confidence_threshold": 0.02,
        "top_k": 5000,
        "nms_threshold": 0.4,
        "keep_top_k": 750,
        "vis_thres": 0.6,  # 可视化/保留结果的置信度阈值
    }

    # 为了兼容原detect.py的参数格式，将字典转为类对象
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Args(**params)
    net, cfg, device = init_detector(args)
    return net, cfg, device, args


def Retinaface_infer(img_path, output_path, net, cfg, device, args, save_img=False):
    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    dets = detect_face(net, cfg, device, img_raw, args)
    # 计算脸型和比例并处理结果
    results = []
    for i, det in enumerate(dets):
        confidence = det[4]
        if confidence < args.vis_thres:
            continue  # 过滤低置信度结果

        # 提取关键点（det[5:15]为5个关键点的坐标）
        landmarks = det[5:15]
        # 计算脸型和比例
        metrics = calculate_face_metrics(landmarks)
        results.append({
            "face_id": i + 1,
            "confidence": round(confidence, 4),
            **metrics
        })

        if save_img:
            # 可视化标注
            x1, y1, x2, y2 = map(int, det[:4])
            cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 画关键点
            for j in range(5):
                x, y = int(landmarks[2 * j]), int(landmarks[2 * j + 1])
                cv2.circle(img_raw, (x, y), 3, (0, 255, 0), -1)
            # 标注脸型
            cv2.putText(img_raw, f"{metrics['face_shape']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 保存结果和输出信息
    if save_img:
        cv2.imwrite(output_path, img_raw)
    # print(f"脸型结果图像已保存至：{output_path}")

    res = results[0]        # 保留第一个人脸，要多个需要修改这里的代码
    output = {
        '脸型': res['face_shape'],
        '宽高比': res['width_height_ratio'],
        '五眼比例': res['five_eye_ratio'],
        '三庭比例': res['three_court_ratio']
    }

    # print("\n人脸分析结果：")
    # for res in results:
    #     print(f"人脸 {res['face_id']}（置信度：{res['confidence']}）:")
    #     print(f"  脸型：{res['face_shape']}")
    #     print(f"  宽高比：{res['width_height_ratio']}")
    #     print(f"  五眼比例：{res['five_eye_ratio']}")
    #     print(f"  三庭比例（中庭/下庭）：{res['three_court_ratio']}\n")

    return output

