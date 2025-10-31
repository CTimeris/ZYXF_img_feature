根据用户图片提取特征：
model_2_column = {
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
"""
    model_name共有以下选项：
    "skin_type", "skin_wrinkles", "age_detect", "hair_type", "hair_color", "emotions", "gender_classify", "cloth_detect"
    "human_beauty", "clip_vit"
"""
