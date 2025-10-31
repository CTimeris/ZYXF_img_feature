import os
import pandas as pd
import torch
import torchvision.transforms as T
from sklearn.metrics import classification_report
from torchvision.transforms.functional import InterpolationMode


def get_img_paths(img_dir):
    """获取目录下所有图片路径"""
    img_paths = []
    for filename in os.listdir(img_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(img_dir, filename)
            img_paths.append(img_path)
    return img_paths


def process_img_beauty(img, input_size=448, max_num=12):
    """human_beauty的输入需要特殊处理"""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)   # 动态分割
    pixel_values = [transform(image) for image in images]   # 对每个子图块应用转换
    pixel_values = torch.stack(pixel_values)                # 拼接为一个batch tensor
    return pixel_values


def build_transform(input_size):
    """将图像转为模型输入要求的格式"""
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """找到最接近原图的子图宽高比"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=12, min_num=1):
    """分割出子图"""
    ori_width, ori_height = img.size  # 原始图像宽高
    aspect_ratio = ori_width / ori_height  # 原始宽高比
    # 生成可能的目标宽高比，确保分割后的块数在min_num, max_num之间
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if max_num >= i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # 与原始宽高比最接近的目标比例
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, ori_width, ori_height, image_size)
    # 计算目标宽高比
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # 缩放到目标宽高，分割成多个image_size x image_size的块
    resized_img = img.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        # 计算每个块的坐标（按网格分割）
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)  # 裁剪出单个块
        processed_images.append(split_img)
    # 可添加缩略图（将原始图像缩放到image_size × image_size作为额外输入）
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = img.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def save_results(results, res_dir):
    """保存输出"""
    output_file = os.path.join(res_dir, "results.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            for k, v in res.items():
                f.write(f"{k}: {v}\n")
            f.write("-------------------------\n")
    print(f"预测结果已保存到 {output_file}")


def test_res(df_labels, df_results, merge_column, column, save_dir):
    """测试结果"""
    labels = df_labels[[merge_column, column]]
    df_pre = df_results[[merge_column, column]].rename(columns={column: f"{column}_pre"})
    df_merge = pd.merge(labels, df_pre, on=merge_column, how='inner')
    # df_merge有三列：图片路径, column_pre, column
    df_merge.to_csv(os.path.join(save_dir, f"{column}_case.csv"), index=False)
    report_str = classification_report(df_merge[column], df_merge[f"{column}_pre"], zero_division=0)
    print(report_str)
