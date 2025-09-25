import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from osgeo import gdal

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img_tensor = torch.from_numpy(
        BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False)
    ).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img_tensor)
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]),
                               mode='bilinear', align_corners=False)
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            prob = torch.sigmoid(output)
            mask = (prob > out_threshold).long().squeeze(1)
    return mask[0].cpu().numpy()


def copy_georef(original_path, target_path):
    orig = gdal.Open(original_path)
    tgt = gdal.Open(target_path, gdal.GA_Update)
    if orig and tgt:
        tgt.SetGeoTransform(orig.GetGeoTransform())
        tgt.SetProjection(orig.GetProjection())
    orig = None
    tgt = None


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./checkpoints/best_model.pth', help='模型文件路径')
    parser.add_argument('--input-folder', '-i', default='data/Test', help='输入图像文件夹 (默认: data/test/images)')
    parser.add_argument('--output-folder', '-o', default='data/output_masks', help='输出掩膜文件夹 (默认: data/test/output)')
    parser.add_argument('--viz', '-v', action='store_true', help='可视化处理过程')
    parser.add_argument('--no-save', '-n', action='store_true', help='不保存掩膜')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='二分类阈值')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='缩放比例 (0-1]')
    parser.add_argument('--bilinear', action='store_true', default=True, help='使用双线性上采样 (默认: True)')
    parser.add_argument('--classes', '-c', type=int, default=2, help='类别数 (默认: 2)')
    parser.add_argument('--use_kan', action='store_true', help='启用 KAN MLP')
    parser.add_argument('--kan_type', choices=['vanilla', 'fast'], default='vanilla', help='KAN 类型')
    parser.add_argument('--kan_dropout', type=float, default=0.1, help='KAN dropout 概率 (默认: 0.1)')
    # 忽略未知参数
    args = parser.parse_known_args()[0]
    return args


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], (list, tuple)):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v
    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not os.path.isdir(args.input_folder):
        logging.error(f'输入文件夹不存在: {args.input_folder}')
        exit(1)
    os.makedirs(args.output_folder, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    try:
        state = torch.load(args.model, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.model, map_location=device)
    except Exception as e:
        logging.error(f'加载模型出错: {e}')
        exit(1)
    mask_values = state.pop('mask_values', [0, 1])
    model = UNet(n_channels=3, n_classes=args.classes,
                 bilinear=args.bilinear,
                 use_kan=args.use_kan,
                 kan_type=args.kan_type,
                 dropout=args.kan_dropout)
    model.load_state_dict(state)
    model.to(device=device, memory_format=torch.channels_last)
    model.eval()

    files = sorted(f for f in os.listdir(args.input_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')))
    for fname in files:
        src = os.path.join(args.input_folder, fname)
        logging.info(f'Processing {src}')
        try:
            img = Image.open(src).convert('RGB')
        except Exception as e:
            logging.error(f'无法打开 {src}: {e}')
            continue

        mask = predict_img(model, img, device,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold)
        if not args.no_save:
            base = os.path.splitext(fname)[0]
            out_file = os.path.join(args.output_folder, f'{base}.tif')
            try:
                result_img = mask_to_image(mask, mask_values)
                result_img.save(out_file)
                logging.info(f'Saved {out_file}')
                copy_georef(src, out_file)
                logging.info(f'Applied georef to {out_file}')
            except Exception as e:
                logging.error(f'写出失败 {out_file}: {e}')

        if args.viz:
            try:
                plot_img_and_mask(img, mask)
            except Exception as e:
                logging.error(f'可视化失败: {e}')
    logging.info('Prediction finished.')
