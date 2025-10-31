from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel, AutoTokenizer
from transformers import CLIPProcessor, CLIPModel
import torch
from utils import process_img_beauty
# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"


class Models:
    def __init__(self, model2path, model2column=None, clip_prompts=None):
        self.model2path = model2path        # 模型名：路径
        self.model2column = model2column    # 模型名：列名
        self.processor_dict = {}
        self.model_dict = {}
        self.clip_prompts = clip_prompts    # clip模型的提示模板

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
            else:
                # 其他模型
                self.processor_dict[model_name] = AutoImageProcessor.from_pretrained(model_path)
                self.model_dict[model_name] = AutoModelForImageClassification.from_pretrained(model_path)
        print("所有模型加载完毕")

    def infer_one_img(self, img):
        """推理一张图片"""
        if not self.model_dict or not self.processor_dict:
            print("先使用.load_model()加载模型")

        results = {}
        # 每个模型分别推理一次图片
        for model_name in self.processor_dict.keys():
            processor = self.processor_dict[model_name]
            model = self.model_dict[model_name]
            if model_name == "human_beauty":
                # 美学大模型
                pixel_values = process_img_beauty(img)
                result = self.human_beauty_infer(pixel_values, processor, model)
                results.update(result)
            elif model_name == "clip_vit":
                # clip大模型
                for column, prompt in self.clip_prompts.items():
                    probs = self.clip_infer(img, prompt, processor, model)
                    probs_flat = probs.squeeze()
                    index = torch.argmax(probs_flat).item()
                    results[column] = index     # 结果保留为prompts中对应的类别下标
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

    def clip_infer(self, image, text, processor, model):
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # 图文相似度
        probs = logits_per_image.softmax(dim=1)  # 标签概率
        return probs

    def human_beauty_infer(self, pixel_values, tokenizer, model):
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

