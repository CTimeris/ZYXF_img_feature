# -*- coding: utf-8 -*-
import gc
from datetime import datetime
from PIL import Image
import torch.cuda
from models import Models, load_Retinaface, Retinaface_infer
import pandas as pd
from utils import save_results, test_res, get_img_paths
import os
import argparse

# 模型名：对应需处理的labels中的列名
model2column = {
        'skin_type': '皮肤质感',
        'skin_wrinkles': '皮肤皱纹',
        'age_detect': '年龄范围',
        'hair_type': '发型',
        'hair_color': '发色',
        'emotions': '表情',
        'gender_classify': '性别',
        'cloth_detect': '衣服类型',
        'human_beauty': None,
        'clip_vit': ['眼镜', '帽子', '首饰'],
    }
# 'Retinaface': ['脸型', '宽高比', '五眼比例', '三庭比例']           显存oom，单独处理

# 模型名：模型路径
model2path = {
        'skin_type': 'model_weights/skin_type',
        'skin_wrinkles': 'model_weights/skin_wrinkles',
        'age_detect': 'model_weights/age_detect',
        'hair_type': 'model_weights/hair_type',
        'hair_color': 'model_weights/hair_color',
        'emotions': 'model_weights/emotions',
        'gender_classify': 'model_weights/gender_classify',
        'cloth_detect': 'model_weights/cloth_detect',
        'human_beauty': 'model_weights/human_beauty',
        'clip_vit': 'model_weights/clip_vit'
}
# clip模型中的提示模板
clip_prompts = {
    '眼镜': [
        'people wearing nearsighted glasses',
        'people wearing sunglasses',
        'people not wearing any glasses'
    ],
    '帽子': [
        'people wearing a hat (cap, beanie, or other headwear)',
        'people with no hat on the head'
    ],
    '首饰': [
        'people wearing accessories such as watches, necklaces, rings, bracelets, earrings, or other jewelry',
        'person with no accessories at all'
    ]
}

# 所有的模型名
model_names = ['skin_type', 'skin_wrinkles', 'age_detect', 'hair_type', 'hair_color', 'emotions', 'gender_classify',
               'cloth_detect', 'human_beauty', 'clip_vit']
# 以及'Retinaface'

# 运行参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--select_models', type=str, default='all',
                    help='用逗号分隔需要跑的模型（如--select_models=skin_type,clip_vit），默认全跑')
parser.add_argument('--img_dir', type=str, default='datasets/test_imgs', help='需要推理的图片路径')
parser.add_argument('--save_dir', type=str, default='res', help='保存结果的目录')
parser.add_argument('--label_file', type=str, default='datasets/labels.txt', help='标签数据')
parser.add_argument('--do_eval', action='store_true', help='是否需要评估')
parser.add_argument('--output_level', type=int, default=2,
                    help='美学大模型输出内容等级，从1到4，1最快，只做整体评分，4最慢输出全部评分')
parser.add_argument('--use_half', action='store_true', help='半精度推理')

parser.add_argument('--no_Retinaface', action='store_true')
parser.add_argument('--save_img', action='store_true', help='是否保存Retinaface检测的图片')
parser.add_argument('--img_output_dir', type=str, default='img_output', help='Retinaface图片输出路径')


def main():
    args = parser.parse_args()
    # ---解析参数---
    if args.select_models != 'none':
        if args.select_models == 'all':
            select_model2path = model2path
            select_model2column = model2column
        else:
            select_models = args.select_models.split(',')
            select_model2path = {k: model2path[k] for k in select_models}
            select_model2column = {k: model2column[k] for k in select_models}
    else:
        select_model2path = {}
        select_model2column = {}
    # 检查保存目录
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    img_output_dir = os.path.join(args.save_dir, args.img_output_dir)
    if args.save_img:
        if not os.path.exists(img_output_dir):  # 检测图片的输出目录
            os.mkdir(img_output_dir)
    batch_size = args.batch_size

    # ---获取图片路径---
    img_dir = args.img_dir  # 图片目录
    img_paths = get_img_paths(img_dir)  # 目录下的所有图片路径
    results = []        # 保存结果的列表
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        result = {'图片路径': img_path}
        results.append(result)

    print(f"{img_dir}目录下共{len(img_paths)}张图片，使用{batch_size}的batch大小推理")

    # ---加载模型---
    all_models = Models(select_model2path, select_model2column, clip_prompts, args.output_level)
    all_models.load_models()
    if not args.no_Retinaface:
        net, cfg, Retinaface_device, Retinaface_args = load_Retinaface()  # 加载Retinaface模型
    time1 = datetime.now()
    print(f"所有模型加载完毕，当前时间: {time1}")
    # 推理
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i: i + batch_size]
        batch_imgs = [Image.open(path).convert('RGB') for path in batch_paths]
        batch_outputs = all_models.infer_imgs(batch_imgs)  # 汇集所有输出的字典
        for j in range(len(batch_paths)):
            results[i + j].update(batch_outputs[j])
            if not args.no_Retinaface:
                img_path = batch_paths[j]
                # 调用原有单张推理函数
                retina_result = Retinaface_infer(
                    img_path, img_output_dir, net, cfg,
                    Retinaface_device, Retinaface_args, args.save_img
                )
                results[i + j].update(retina_result)
        torch.cuda.empty_cache()
        gc.collect()

    time2 = datetime.now()
    print(f"推理完毕，当前时间: {time2}。平均推理时间: {(time2 - time1) / len(img_paths)}")

    # ---保存结果---
    df_results = pd.DataFrame(results)
    img_dir_name = os.path.basename(img_dir)
    df_results.to_csv(os.path.join(args.save_dir, f'{img_dir_name}_results.csv'), index=False)
    # save_results(results, save_dir)       # 一行一行写，保存为txt文件

    # ---评估结果---
    if args.do_eval:
        label_file = args.label_file
        df_labels = pd.read_csv(label_file, sep=',', header=0)
        merge_column = '图片路径'  # 根据图片路径这一列合并
        columns = select_model2column.values()
        for column in columns:
            if column and column is not list:
                test_res(df_labels, df_results, merge_column, column, args.save_dir)    # 标签，结果，合并列名，预测列名，保存目录

    print('done')


if __name__ == "__main__":
    main()

