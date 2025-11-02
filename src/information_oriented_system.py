"""
信息化导向系统 v2.0 - Information-Oriented System (重构版)

采用信息粒子化理论：
1. 信息粒子提取：将数据粒子化为最小信息单元
2. 信息组构建：聚合粒子形成语义单元
3. 球面结构化：纯数学映射到球面空间
4. 无损重构：完美恢复原始数据

核心创新：
- 完全透明：纯数学方法，无神经网络
- 12维特征：明确的物理意义（借鉴时间集维度设计）
- SIF值：Structure-Information-Function综合评估
- 完美重构：MSE=0，无信息损失

作者：北京求一数生科技中心
版本：2.0.0 (理论重构版)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import time

from information_particle_system import (
    InformationParticle, InformationGroup,
    InformationParticleExtractor, InformationGroupBuilder,
    LosslessReconstructor, PureMathematicalSphereMapper
)


class InformationOrientedSystemV2(nn.Module):
    """
    信息化导向系统 v2.0
    
    完整流程：
    1. 粒子化 → 信息粒子（12维特征）
    2. 聚合 → 信息组（语义单元）
    3. 球面映射 → 球面坐标(r, θ, φ)
    4. 重构 → 无损恢复原始数据
    
    特点：
    - 纯数学方法（无神经网络）
    - 完全透明可解释
    - 完美重构（MSE=0）
    """
    
    def __init__(
        self, 
        particle_size: int = 28,
        similarity_threshold: float = 0.5,
        max_group_size: int = 10,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.device = device
        
        # 信息粒子提取器（纯规则）
        self.particle_extractor = InformationParticleExtractor(
            particle_size=particle_size,
            device=device
        )
        
        # 信息组构建器（纯规则）
        self.group_builder = InformationGroupBuilder(
            similarity_threshold=similarity_threshold,
            max_group_size=max_group_size
        )
        
        # 球面映射器（纯数学）
        self.sphere_mapper = PureMathematicalSphereMapper()
        
        # 无损重构器（直接提取）
        self.reconstructor = LosslessReconstructor()
        
        self.system_info = {
            'version': '2.0.0',
            'author': '北京求一数生科技中心',
            'theory': '信息粒子化理论',
            'features': '12维特征（时间4+空间3+结构4+SIF1）'
        }
    
    def forward(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        完整的信息化处理流程
        
        Args:
            input_data: [seq_len, feature_dim] 或 [batch, seq_len, feature_dim]
        
        Returns:
            包含粒子、组、球面坐标等的完整结果
        """
        # 处理批次维度
        if input_data.dim() == 3:
            batch_size = input_data.shape[0]
            # 暂时只处理第一个样本
            input_data = input_data[0]
        else:
            batch_size = 1
        
        # 1. 信息粒子化
        particles = self.particle_extractor.extract(input_data)
        
        # 2. 信息组构建
        groups = self.group_builder.build_groups(particles)
        
        # 3. 球面映射
        particle_sphere_coords = self.sphere_mapper.map_all_particles(particles)
        group_sphere_coords = self.sphere_mapper.map_all_groups(groups)
        
        return {
            'success': True,
            'particles': particles,
            'groups': groups,
            'particle_sphere_coords': particle_sphere_coords,
            'group_sphere_coords': group_sphere_coords,
            'num_particles': len(particles),
            'num_groups': len(groups),
            'avg_sif': sum(p.sif_value for p in particles) / len(particles) if particles else 0,
            'system_info': self.system_info
        }
    
    def reconstruct(self, output: Dict[str, Any]) -> torch.Tensor:
        """
        无损重构原始数据
        
        Args:
            output: forward方法的输出
        
        Returns:
            重构的数据
        """
        particles = output.get('particles', [])
        return self.reconstructor.reconstruct_from_particles(particles)
    
    def get_particle_features(self, output: Dict[str, Any]) -> torch.Tensor:
        """
        获取所有粒子的12维特征向量
        
        Args:
            output: forward方法的输出
        
        Returns:
            [num_particles, 12] 特征矩阵
        """
        particles = output.get('particles', [])
        if not particles:
            return None
        
        features = torch.stack([p.to_vector() for p in particles])
        return features
    
    def get_sphere_representation(self, output: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        获取球面表示
        
        Args:
            output: forward方法的输出
        
        Returns:
            球面坐标和笛卡尔坐标
        """
        particle_coords = output.get('particle_sphere_coords', [])
        
        if not particle_coords:
            return None
        
        # 提取球面坐标
        r_values = [c['spherical']['r'] for c in particle_coords]
        theta_values = [c['spherical']['theta'] for c in particle_coords]
        phi_values = [c['spherical']['phi'] for c in particle_coords]
        
        # 提取笛卡尔坐标
        x_values = [c['cartesian']['x'] for c in particle_coords]
        y_values = [c['cartesian']['y'] for c in particle_coords]
        z_values = [c['cartesian']['z'] for c in particle_coords]
        
        return {
            'spherical': {
                'r': torch.tensor(r_values),
                'theta': torch.tensor(theta_values),
                'phi': torch.tensor(phi_values)
            },
            'cartesian': {
                'x': torch.tensor(x_values),
                'y': torch.tensor(y_values),
                'z': torch.tensor(z_values)
            }
        }
    
    def print_system_info(self):
        """打印系统信息"""
        print("\n" + "="*70)
        print("  信息化导向系统 v2.0")
        print("  Information-Oriented System v2.0")
        print("="*70)
        print(f"  版本: {self.system_info['version']}")
        print(f"  作者: {self.system_info['author']}")
        print(f"  理论: {self.system_info['theory']}")
        print(f"  特征: {self.system_info['features']}")
        print("="*70)


def test_system_on_mnist():
    """
    在类似MNIST的数据上测试系统
    """
    print("\n" + "="*70)
    print("  信息化导向系统 v2.0 测试")
    print("  在类似MNIST数据上验证")
    print("="*70)
    
    # 创建系统
    system = InformationOrientedSystemV2(
        particle_size=28,
        similarity_threshold=0.5
    )
    
    # 打印系统信息
    system.print_system_info()
    
    # 创建测试数据（28x28，类似MNIST）
    print("\n📊 创建测试数据...")
    test_data = torch.randn(28, 28)
    print(f"   数据形状: {test_data.shape}")
    
    # 前向处理
    print("\n" + "="*70)
    print("\n🔄 执行信息化处理...")
    start_time = time.time()
    output = system.forward(test_data)
    process_time = (time.time() - start_time) * 1000
    
    # 重构
    print("\n" + "="*70)
    reconstructed = system.reconstruct(output)
    
    # 验证
    print("\n" + "="*70)
    print("\n🔍 验证结果...")
    mse = F.mse_loss(reconstructed, test_data).item()
    cos_sim = F.cosine_similarity(
        reconstructed.flatten(), 
        test_data.flatten(), 
        dim=0
    ).item()
    
    print(f"\n✅ 处理完成:")
    print(f"   处理时间: {process_time:.2f} ms")
    print(f"   生成粒子数: {output['num_particles']}")
    print(f"   生成信息组数: {output['num_groups']}")
    print(f"   平均SIF值: {output['avg_sif']:.4f}")
    
    print(f"\n✅ 重构质量:")
    print(f"   MSE: {mse:.10f}")
    print(f"   Cosine Similarity: {cos_sim:.10f}")
    
    if mse < 1e-6:
        print(f"   ✅ 完美重构！（MSE ≈ 0）")
    else:
        print(f"   ⚠️  重构有误差")
    
    # 获取特征
    features = system.get_particle_features(output)
    print(f"\n📊 粒子特征矩阵: {features.shape}")
    
    # 获取球面表示
    sphere_repr = system.get_sphere_representation(output)
    print(f"\n🌐 球面表示:")
    print(f"   径向r范围: [{sphere_repr['spherical']['r'].min():.3f}, {sphere_repr['spherical']['r'].max():.3f}]")
    print(f"   极角θ范围: [{sphere_repr['spherical']['theta'].min():.3f}, {sphere_repr['spherical']['theta'].max():.3f}]")
    print(f"   方位角φ范围: [{sphere_repr['spherical']['phi'].min():.3f}, {sphere_repr['spherical']['phi'].max():.3f}]")
    
    print("\n" + "="*70)
    print("  测试完成！")
    print("="*70)
    
    return {
        'system': system,
        'output': output,
        'reconstructed': reconstructed,
        'mse': mse,
        'cosine_similarity': cos_sim,
        'features': features,
        'sphere_repr': sphere_repr
    }


if __name__ == '__main__':
    # 运行测试
    test_system_on_mnist()

