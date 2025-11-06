from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from .data import cfg_mnet, cfg_re50
from .layers.functions.prior_box import PriorBox
from .utils.nms.py_cpu_nms import py_cpu_nms
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm


# 保留原有工具函数
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def init_detector(args):
    """初始化人脸检测器并返回模型和配置"""
    cfg = cfg_mnet if args.network == "mobile0.25" else cfg_re50
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    return net, cfg, device


def detect_face(net, cfg, device, img_raw, args):
    """
    输入图像，返回检测到的人脸框和关键点
    :param img_raw: cv2读取的BGR图像
    :return: dets: 检测结果数组，每个元素包含[框坐标, 置信度, 5个关键点坐标]
    """
    resize = 1
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).to(device)

    # 图像预处理
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    # 模型推理
    loc, conf, landms = net(img)

    # 解码框和关键点
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)
    boxes = decode(loc.data.squeeze(0), priors.data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()

    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]  # 人脸置信度

    # 关键点解码
    scale1 = torch.Tensor([img.shape[3], img.shape[2]] * 5).to(device)
    landms = decode_landm(landms.data.squeeze(0), priors.data, cfg['variance'])
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # 筛选高置信度结果
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # NMS处理
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    dets = dets[keep, :][:args.keep_top_k, :]
    landms = landms[keep][:args.keep_top_k, :]

    # 合并框和关键点（最终shape: [N, 15]）
    dets = np.concatenate((dets, landms), axis=1)
    return dets