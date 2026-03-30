#!/usr/bin/env python3
"""
半监督预测脚本 - 使用多个分割模型进行投票式标注

通过多个训练好的分割模型对未标注的数据进行预测，并使用投票机制生成标签。
支持的模型:
- csv_seg_outputs_efficientnet-b7
- csv_seg_outputs_inceptionresnetv2
- csv_seg_outputs_resnet152
- csv_seg_outputs_se_resnet152
- csv_seg_outputs_mit_b5

投票策略:
- 对于分割: 使用像素级多数投票
- 对于分类: 使用模型输出的概率加权投票
"""

import os
import sys
import glob
import argparse
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple
import json

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from csv_model import CSVModel


class ModelEnsemble:
    """多模型集成类，用于投票预测"""
    
    def __init__(self, model_configs: List[Dict], device: str = 'cuda'):
        """
        Args:
            model_configs: 模型配置列表，每个包含:
                - checkpoint: 模型检查点路径
                - encoder: 编码器名称
                - resize_target: 目标尺寸
            device: 运行设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.model_names = []
        
        print(f"Loading {len(model_configs)} models for ensemble prediction...")
        
        for i, config in enumerate(model_configs):
            try:
                # 创建模型
                model = CSVModel(
                    encoder_name=config['encoder'],
                    encoder_weights=None,
                    num_seg_classes=3,
                    num_cls_classes=2,
                    use_classification=False  # 只使用分割
                ).to(self.device)
                
                # 加载检查点
                ckpt_path = config['checkpoint']
                if not os.path.exists(ckpt_path):
                    print(f"Warning: Checkpoint not found: {ckpt_path}, skipping...")
                    continue
                
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                
                if isinstance(ckpt, dict):
                    if "model_state_dict" in ckpt:
                        state = ckpt["model_state_dict"]
                    elif "state_dict" in ckpt:
                        state = ckpt["state_dict"]
                    elif "model" in ckpt:
                        state = ckpt["model"]
                    else:
                        state = ckpt
                else:
                    state = ckpt
                
                model.load_state_dict(state, strict=True)
                model.eval()
                
                self.models.append({
                    'model': model,
                    'resize_target': config['resize_target'],
                    'encoder': config['encoder']
                })
                self.model_names.append(os.path.basename(os.path.dirname(ckpt_path)))
                
                print(f"✓ Loaded model {i+1}/{len(model_configs)}: {config['encoder']}")
                
            except Exception as e:
                print(f"✗ Failed to load model {config['encoder']}: {e}")
                continue
        
        if len(self.models) == 0:
            raise ValueError("No models loaded successfully!")
        
        print(f"\n✓ Total models loaded: {len(self.models)}")
    
    def predict_single_view(self, image_tensor: torch.Tensor, 
                          original_shape: Tuple[int, int]) -> np.ndarray:
        """
        对单个视图进行集成预测
        
        Args:
            image_tensor: 输入图像 [3, H, W]
            original_shape: 原始图像尺寸 (H, W)
        
        Returns:
            预测的分割掩码 (H, W)，值为 0, 128, 255
        """
        predictions = []
        
        with torch.no_grad():
            for model_info in self.models:
                model = model_info['model']
                resize_target = model_info['resize_target']
                
                # 添加batch维度并移到设备
                x = image_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]
                
                # 调整大小到训练尺寸
                x_resized = F.interpolate(
                    x, 
                    (resize_target, resize_target), 
                    mode="bilinear", 
                    align_corners=False
                )
                
                # 前向传播
                outputs = model(x_resized)
                seg_logits = outputs['seg_logits']
                
                # 上采样回原始尺寸
                seg_logits_up = F.interpolate(
                    seg_logits, 
                    original_shape, 
                    mode="bilinear", 
                    align_corners=False
                )
                
                # 获取预测类别 [1, H, W]
                pred = torch.argmax(seg_logits_up, dim=1).squeeze(0).cpu().numpy()
                predictions.append(pred)
        
        # 投票: 对每个像素取众数
        predictions = np.stack(predictions, axis=0)  # [N_models, H, W]
        
        # 使用numpy的bincount进行投票
        voted_pred = np.zeros(original_shape, dtype=np.uint8)
        for i in range(original_shape[0]):
            for j in range(original_shape[1]):
                pixel_votes = predictions[:, i, j]
                # 统计每个类别的票数
                voted_pred[i, j] = np.bincount(pixel_votes).argmax()
        
        # 转换回 0, 128, 255 格式
        result = np.zeros_like(voted_pred)
        result[voted_pred == 1] = 128  # Plaque
        result[voted_pred == 2] = 255  # Vessel
        
        return result


def load_and_preprocess_image(image_path: str, view: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    加载并预处理图像
    
    Args:
        image_path: 图像文件路径
        view: 'long' or 'trans'
    
    Returns:
        image_tensor: 预处理后的图像张量 [3, H, W]
        original_shape: 原始图像尺寸 (H, W)
    """
    with h5py.File(image_path, "r") as f:
        if view == 'long':
            image = f["long_img"][:]
        else:
            image = f["trans_img"][:]
    
    original_shape = image.shape
    
    # 转换为3通道
    image_3ch = np.stack([image, image, image], axis=-1).astype(np.float32)
    
    # 归一化到 [0, 255]
    if image_3ch.max() <= 1.0:
        image_3ch = image_3ch * 255.0
    image_3ch = image_3ch.astype(np.uint8)
    
    # 转换为torch张量: [3, H, W]
    image_tensor = torch.from_numpy(image_3ch.transpose(2, 0, 1)).float()
    
    # 归一化到 [0, 1]
    image_tensor = image_tensor / 255.0
    
    # 应用ImageNet标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor, original_shape


def predict_unlabeled_data(
    ensemble: ModelEnsemble,
    images_dir: str,
    output_dir: str,
    start_idx: int = 200,
    end_idx: int = 1000,
    cls_value: int = 0  # 默认分类标签 (0: low risk, 1: high risk)
):
    """
    对未标注数据进行预测并生成伪标签
    
    Args:
        ensemble: 模型集成对象
        images_dir: 未标注图像目录
        output_dir: 输出标签目录
        start_idx: 起始索引 (默认200，未标注数据起点)
        end_idx: 结束索引 (默认1000)
        cls_value: 默认分类标签值
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting semi-supervised labeling...")
    print(f"Images dir: {images_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Range: {start_idx:04d} - {end_idx-1:04d} ({end_idx - start_idx} cases)")
    print(f"Default classification label: {cls_value}")
    print(f"{'='*60}\n")
    
    success_count = 0
    failed_count = 0
    failed_cases = []
    
    for case_id in tqdm(range(start_idx, end_idx), desc="Labeling"):
        image_path = os.path.join(images_dir, f"{case_id:04d}.h5")
        output_path = os.path.join(output_dir, f"{case_id:04d}_label.h5")
        
        # 跳过已存在的标签
        if os.path.exists(output_path):
            continue
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            failed_count += 1
            failed_cases.append(case_id)
            continue
        
        try:
            # 预测长轴视图
            long_tensor, long_shape = load_and_preprocess_image(image_path, 'long')
            long_mask = ensemble.predict_single_view(long_tensor, long_shape)
            
            # 预测横轴视图
            trans_tensor, trans_shape = load_and_preprocess_image(image_path, 'trans')
            trans_mask = ensemble.predict_single_view(trans_tensor, trans_shape)
            
            # 保存为h5文件
            with h5py.File(output_path, "w") as hf:
                hf.create_dataset("long_mask", data=long_mask, compression="gzip")
                hf.create_dataset("trans_mask", data=trans_mask, compression="gzip")
                hf.create_dataset("cls", data=cls_value)
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing case {case_id:04d}: {e}")
            failed_count += 1
            failed_cases.append(case_id)
            continue
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"Labeling completed!")
    print(f"✓ Successfully labeled: {success_count}")
    print(f"✗ Failed: {failed_count}")
    if failed_cases:
        print(f"Failed cases: {failed_cases[:10]}{'...' if len(failed_cases) > 10 else ''}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # 保存统计信息
    stats = {
        'total_cases': end_idx - start_idx,
        'success_count': success_count,
        'failed_count': failed_count,
        'failed_cases': failed_cases,
        'models_used': ensemble.model_names,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'cls_value': cls_value
    }
    
    stats_path = os.path.join(output_dir, 'labeling_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to: {stats_path}")


def discover_models(base_dir: str) -> List[Dict]:
    """
    自动发现所有可用的分割模型
    
    Args:
        base_dir: 项目根目录
    
    Returns:
        模型配置列表
    """
    model_configs = []
    
    # 模型目录到编码器名称的映射
    encoder_map = {
        'csv_seg_outputs_efficientnet-b7': ('efficientnet-b7', 512),
        'csv_seg_outputs_inceptionresnetv2': ('inceptionresnetv2', 512),
        'csv_seg_outputs_resnet152': ('resnet152', 512),
        'csv_seg_outputs_se_resnet152': ('se_resnet152', 512),
        'csv_seg_outputs_mit_b5': ('mit_b5', 512)
    }
    
    for model_dir, (encoder, resize_target) in encoder_map.items():
        model_path = os.path.join(base_dir, model_dir)
        checkpoint_path = os.path.join(model_path, 'best_model.pth')
        
        if os.path.exists(checkpoint_path):
            model_configs.append({
                'checkpoint': checkpoint_path,
                'encoder': encoder,
                'resize_target': resize_target
            })
            print(f"Found model: {model_dir}")
    
    return model_configs


def main():
    parser = argparse.ArgumentParser(
        description="Semi-supervised labeling using ensemble voting"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="./data/images",
        help="Directory containing unlabeled images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/pseudo_labels",
        help="Output directory for generated labels"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=".",
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=200,
        help="Start index for unlabeled data"
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=1000,
        help="End index for unlabeled data"
    )
    parser.add_argument(
        "--cls-value",
        type=int,
        default=0,
        choices=[0, 1],
        help="Default classification value (0: low risk, 1: high risk)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=['cuda', 'cpu'],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=None,
        help="Specific model directories to use (optional)"
    )
    
    args = parser.parse_args()
    
    # 自动发现或使用指定的模型
    if args.models:
        model_configs = []
        encoder_map = {
            'efficientnet-b7': 512,
            'inceptionresnetv2': 512,
            'resnet152': 512,
            'se_resnet152': 512,
            'mit_b5': 512
        }
        for model_path in args.models:
            checkpoint = os.path.join(model_path, 'best_model.pth')
            # 从路径中提取编码器名称
            for encoder in encoder_map:
                if encoder in model_path:
                    model_configs.append({
                        'checkpoint': checkpoint,
                        'encoder': encoder,
                        'resize_target': encoder_map[encoder]
                    })
                    break
    else:
        print("Auto-discovering models...")
        model_configs = discover_models(args.model_dir)
    
    if not model_configs:
        print("Error: No models found!")
        return
    
    print(f"\nFound {len(model_configs)} models")
    
    # 创建模型集成
    ensemble = ModelEnsemble(model_configs, device=args.device)
    
    # 执行预测
    predict_unlabeled_data(
        ensemble=ensemble,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        cls_value=args.cls_value
    )


if __name__ == "__main__":
    main()
