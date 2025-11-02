"""
信息粒子系统 - Information Particle System

核心理念：信息粒子化（类似像素点）
- 信息元 = 信息的最小单元（粒子/像素点）
- 12维特征结构（借鉴时间集维度系统设计）
- 纯数学方法（无神经网络）
- 完全透明可解释

理论基础：
1. 信息粒子化：将连续数据离散化为独立的信息单元
2. 五维认知框架：点→面→立体→时间→时间集
3. SIF值：Structure-Information-Function（结构-信息-功能）

作者：北京求一数生科技中心
版本：2.0.0 (理论重构版)
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class InformationParticle:
    """
    信息粒子 - 信息的最小单元（像素点）
    
    核心思想：
    - 每个粒子是信息的基本单元，类似图像的像素点
    - 有明确的12维特征描述其属性
    - 保留原始内容以实现无损重构
    
    12维特征结构（借鉴时间集维度系统）：
    
    [时间维度 4维] - 描述信息的时间属性
    1. inner_time: 内在时间（序列中的相对位置，主观时间）
    2. outer_time: 外部时间（绝对时间戳，客观时间）
    3. time_flow: 时间流速（信息的变化率）
    4. current_time: 当前状态时间
    
    [空间维度 3维] - 描述信息的空间属性
    5. spatial_x: X坐标（序列位置，点维度）
    6. spatial_y: Y坐标（内容均值，面维度）
    7. spatial_z: Z坐标（内容方差，立体维度）
    
    [结构维度 4维] - 描述信息的内在结构
    8. density: 信息密度（内容丰富度，非零元素比例）
    9. connectivity: 信息连接度（与其他粒子的关联）
    10. stability: 信息稳定性（确定性，方差的倒数）
    11. energy: 信息能量（重要性/活跃度，L2范数）
    
    [综合指标 1维]
    12. sif_value: Structure-Information-Function值（综合质量评估）
    """
    
    # === 12维核心特征 ===
    # 时间维度 [4维]
    inner_time: float      # 内在时间（序列索引归一化到[0,1]）
    outer_time: float      # 外部时间（绝对时间戳）
    time_flow: float       # 时间流速（变化率）
    current_time: float    # 当前状态时间
    
    # 空间维度 [3维]
    spatial_x: float       # X坐标（序列位置归一化）
    spatial_y: float       # Y坐标（内容平均值）
    spatial_z: float       # Z坐标（内容方差，反映立体复杂度）
    
    # 结构维度 [4维]
    density: float         # 信息密度（非零元素比例）
    connectivity: float    # 连接度（与相邻粒子的相似度）
    stability: float       # 稳定性（1 / (1 + variance)）
    energy: float          # 能量（L2范数归一化）
    
    # 综合指标 [1维]
    sif_value: float       # Structure-Information-Function值
    
    # === 原始内容（用于无损重构）===
    raw_content: torch.Tensor  # 原始数据片段
    sequence_index: int        # 在序列中的索引位置
    
    # === 可选属性 ===
    semantic_tag: Optional[str] = None  # 语义标签（可选）
    metadata: Optional[Dict[str, Any]] = None  # 元数据（可选）
    
    def to_vector(self) -> torch.Tensor:
        """
        转换为12维特征向量
        
        Returns:
            [12] 维度的特征向量
        """
        return torch.tensor([
            self.inner_time, self.outer_time, 
            self.time_flow, self.current_time,
            self.spatial_x, self.spatial_y, self.spatial_z,
            self.density, self.connectivity, 
            self.stability, self.energy,
            self.sif_value
        ], dtype=torch.float32)
    
    def get_sphere_coordinates(self) -> Tuple[float, float, float]:
        """
        获取球面坐标表示
        
        使用纯数学公式将12维特征映射到球面坐标(r, θ, φ)
        
        Returns:
            (r, theta, phi) 球面坐标
        """
        # 径向r：反映抽象层次（稳定性和能量的综合）
        r = 0.5 * self.stability + 0.5 * self.energy
        r = max(0.1, min(r, 1.0))  # 截断到[0.1, 1.0]
        
        # 极角θ：反映空间位置（0到π）
        theta = np.pi * (0.5 * self.spatial_x + 0.5 * np.tanh(self.spatial_y))
        theta = max(0, min(theta, np.pi))
        
        # 方位角φ：反映时间和关联性（0到2π）
        phi = 2 * np.pi * (0.6 * self.inner_time + 0.4 * self.connectivity)
        phi = phi % (2 * np.pi)  # 确保在[0, 2π]
        
        return (r, theta, phi)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"InformationParticle(idx={self.sequence_index}, "
                f"sif={self.sif_value:.3f}, "
                f"energy={self.energy:.3f}, "
                f"stability={self.stability:.3f})")


@dataclass
class InformationGroup:
    """
    信息组 - 语义完整的信息单元
    
    由多个信息粒子聚合而成，代表一个语义完整的概念
    """
    
    particles: List[InformationParticle]  # 组成此组的粒子列表
    group_id: int                          # 组ID
    
    # 聚合特征（从粒子计算而来）
    centroid_features: torch.Tensor = None  # 质心特征（12维）
    aggregated_content: torch.Tensor = None # 聚合内容
    
    # 组级别的属性
    group_sif: float = 0.0                 # 组的SIF值
    group_stability: float = 0.0           # 组的稳定性
    group_size: int = 0                    # 组的大小
    
    def __post_init__(self):
        """初始化后计算聚合特征"""
        if self.particles:
            self.group_size = len(self.particles)
            self._compute_aggregated_features()
    
    def _compute_aggregated_features(self):
        """计算聚合特征"""
        if not self.particles:
            return
        
        # 计算质心（所有粒子12维特征的平均）
        particle_vectors = torch.stack([p.to_vector() for p in self.particles])
        self.centroid_features = particle_vectors.mean(dim=0)
        
        # 聚合内容（拼接所有raw_content）
        contents = [p.raw_content for p in self.particles]
        self.aggregated_content = torch.cat(contents, dim=0)
        
        # 计算组级别的SIF（平均）
        self.group_sif = sum(p.sif_value for p in self.particles) / len(self.particles)
        
        # 计算组级别的稳定性（平均）
        self.group_stability = sum(p.stability for p in self.particles) / len(self.particles)
    
    def get_sphere_coordinates(self) -> Tuple[float, float, float]:
        """
        获取组的球面坐标（使用质心）
        
        Returns:
            (r, theta, phi) 球面坐标
        """
        # 使用质心特征计算
        stability = self.centroid_features[9].item()
        energy = self.centroid_features[10].item()
        spatial_x = self.centroid_features[4].item()
        spatial_y = self.centroid_features[5].item()
        inner_time = self.centroid_features[0].item()
        connectivity = self.centroid_features[8].item()
        
        # 径向
        r = 0.5 * stability + 0.5 * energy
        r = max(0.1, min(r, 1.0))
        
        # 极角
        theta = np.pi * (0.5 * spatial_x + 0.5 * np.tanh(spatial_y))
        theta = max(0, min(theta, np.pi))
        
        # 方位角
        phi = 2 * np.pi * (0.6 * inner_time + 0.4 * connectivity)
        phi = phi % (2 * np.pi)
        
        return (r, theta, phi)


class InformationParticleExtractor:
    """
    信息粒子提取器 - 纯规则方法，无神经网络
    
    核心功能：
    1. 将原始数据序列分割为信息粒子
    2. 为每个粒子计算12维特征（纯数学统计）
    3. 计算粒子间的连接度
    4. 计算SIF值（Structure-Information-Function）
    
    设计原则：
    - 完全透明：所有计算都是明确的数学公式
    - 无损分割：所有粒子的raw_content拼接后等于原始数据
    - 独立性：每个粒子都是独立的信息单元
    """
    
    def __init__(self, particle_size: int = 28, device: str = 'cpu'):
        """
        Args:
            particle_size: 每个粒子包含的数据点数量（默认28，适合MNIST的28行）
            device: 计算设备
        """
        self.particle_size = particle_size
        self.device = device
        
        # SIF计算的权重（可调整）
        self.sif_weights = {
            'structure': 0.3,    # 结构权重
            'information': 0.4,  # 信息权重
            'function': 0.3      # 功能权重
        }
    
    def extract(self, data: torch.Tensor) -> List[InformationParticle]:
        """
        将数据粒子化
        
        数学原理：
        1. 固定窗口分割（保证无损）
        2. 每个粒子计算12维特征（纯统计方法）
        3. 计算粒子间连接度（相似度）
        4. 计算SIF值（综合评估）
        
        Args:
            data: 输入数据，形状为 [seq_len, feature_dim] 或 [batch, seq_len, feature_dim]
        
        Returns:
            信息粒子列表
        """
        # 处理批次维度
        if data.dim() == 3:
            # [batch, seq_len, feature_dim] -> [seq_len, feature_dim]
            # 取第一个样本
            data = data[0]
        
        # 确保在正确设备上
        if data.device.type != self.device:
            data = data.to(self.device)
        
        seq_len = data.shape[0]
        particles = []
        
        # 固定窗口分割
        num_particles = math.ceil(seq_len / self.particle_size)
        
        print(f"\n🔬 信息粒子化开始...")
        print(f"   输入数据形状: {data.shape}")
        print(f"   粒子大小: {self.particle_size}")
        print(f"   预计粒子数: {num_particles}")
        
        for i in range(num_particles):
            start_idx = i * self.particle_size
            end_idx = min(start_idx + self.particle_size, seq_len)
            
            # 提取原始内容
            raw_content = data[start_idx:end_idx].clone()
            
            # 计算12维特征（纯数学方法）
            particle = self._compute_particle_features(
                raw_content=raw_content,
                sequence_index=i,
                total_particles=num_particles
            )
            
            particles.append(particle)
        
        # 计算粒子间的connectivity（第二遍）
        self._compute_connectivity(particles)
        
        print(f"✅ 粒子化完成，生成 {len(particles)} 个信息粒子")
        print(f"   平均SIF值: {sum(p.sif_value for p in particles) / len(particles):.4f}")
        print(f"   平均能量: {sum(p.energy for p in particles) / len(particles):.4f}")
        
        return particles
    
    def _compute_particle_features(
        self, 
        raw_content: torch.Tensor,
        sequence_index: int,
        total_particles: int
    ) -> InformationParticle:
        """
        纯数学方法计算12维特征
        
        每个特征都有明确的数学定义和物理意义
        
        Args:
            raw_content: 原始数据片段 [chunk_size, feature_dim]
            sequence_index: 序列索引
            total_particles: 总粒子数
        
        Returns:
            InformationParticle
        """
        # === 时间维度 [4维] ===
        # 1. 内在时间：在序列中的相对位置（归一化到[0,1]）
        inner_time = sequence_index / max(total_particles - 1, 1)
        
        # 2. 外部时间：绝对时间戳（当前时间）
        outer_time = time.time()
        
        # 3. 时间流速：暂时设为1.0（可以后续根据变化率计算）
        time_flow = 1.0
        
        # 4. 当前状态时间：与外部时间相同
        current_time = outer_time
        
        # === 空间维度 [3维] ===
        # 5. spatial_x: 序列位置（归一化）
        spatial_x = inner_time  # 与inner_time相同，表示在序列中的位置
        
        # 6. spatial_y: 内容均值（反映内容的平均水平）
        spatial_y = raw_content.mean().item()
        
        # 7. spatial_z: 内容方差（反映内容的复杂度/立体性）
        spatial_z = raw_content.std().item()
        
        # === 结构维度 [4维] ===
        # 8. density: 信息密度（非零元素的比例）
        density = (raw_content != 0).float().mean().item()
        
        # 9. connectivity: 连接度（先设为0，后续计算）
        connectivity = 0.0  # 需要在所有粒子创建后计算
        
        # 10. stability: 稳定性（方差的倒数，方差小则稳定性高）
        variance = raw_content.var().item()
        stability = 1.0 / (1.0 + variance)  # 加1避免除零
        
        # 11. energy: 能量（L2范数，归一化到每个元素）
        energy = raw_content.norm().item() / max(raw_content.numel(), 1)
        
        # === 综合指标 [1维] ===
        # 12. sif_value: 暂时计算（connectivity为0），后续更新
        sif_value = self._compute_sif(
            density=density,
            connectivity=connectivity,
            stability=stability,
            energy=energy,
            spatial_z=spatial_z
        )
        
        return InformationParticle(
            inner_time=inner_time,
            outer_time=outer_time,
            time_flow=time_flow,
            current_time=current_time,
            spatial_x=spatial_x,
            spatial_y=spatial_y,
            spatial_z=spatial_z,
            density=density,
            connectivity=connectivity,
            stability=stability,
            energy=energy,
            sif_value=sif_value,
            raw_content=raw_content,
            sequence_index=sequence_index
        )
    
    def _compute_sif(
        self, 
        density: float,
        connectivity: float,
        stability: float,
        energy: float,
        spatial_z: float
    ) -> float:
        """
        计算SIF值（Structure-Information-Function）
        
        数学定义：
        SIF = α·Structure + β·Information + γ·Function
        
        其中：
        - Structure: 结构完整性（基于空间方差和稳定性）
        - Information: 信息丰富度（基于密度和能量）
        - Function: 功能性（基于连接度）
        - α, β, γ 为权重，满足 α + β + γ = 1
        
        Args:
            density: 信息密度
            connectivity: 连接度
            stability: 稳定性
            energy: 能量
            spatial_z: 空间方差
        
        Returns:
            SIF值，范围[0, 1]
        """
        # Structure: 基于空间方差和稳定性
        # 方差大 → 结构复杂，稳定性高 → 结构好
        structure_score = 0.5 * min(spatial_z, 1.0) + 0.5 * stability
        
        # Information: 基于密度和能量
        # 密度高、能量高 → 信息丰富
        information_score = 0.6 * density + 0.4 * min(energy, 1.0)
        
        # Function: 基于连接度
        # 连接度高 → 功能性强
        function_score = connectivity
        
        # 综合SIF值（使用配置的权重）
        sif = (self.sif_weights['structure'] * structure_score +
               self.sif_weights['information'] * information_score +
               self.sif_weights['function'] * function_score)
        
        # 确保在[0, 1]范围内
        return max(0.0, min(sif, 1.0))
    
    def _compute_connectivity(self, particles: List[InformationParticle]):
        """
        计算粒子间的连接度（相似度）
        
        策略：
        - 第一个粒子：与下一个粒子的相似度
        - 中间粒子：与前后粒子的平均相似度
        - 最后一个粒子：与前一个粒子的相似度
        
        Args:
            particles: 粒子列表（会被原地修改）
        """
        if len(particles) <= 1:
            return
        
        print(f"   计算粒子连接度...")
        
        for i in range(len(particles)):
            if i == 0:
                # 第一个粒子：只与下一个比较
                sim = self._cosine_similarity(
                    particles[i].raw_content,
                    particles[i+1].raw_content
                )
                particles[i].connectivity = sim
                
            elif i == len(particles) - 1:
                # 最后一个粒子：只与前一个比较
                sim = self._cosine_similarity(
                    particles[i].raw_content,
                    particles[i-1].raw_content
                )
                particles[i].connectivity = sim
                
            else:
                # 中间粒子：与前后的平均相似度
                sim_prev = self._cosine_similarity(
                    particles[i].raw_content,
                    particles[i-1].raw_content
                )
                sim_next = self._cosine_similarity(
                    particles[i].raw_content,
                    particles[i+1].raw_content
                )
                particles[i].connectivity = (sim_prev + sim_next) / 2
            
            # 重新计算SIF（现在包含connectivity）
            particles[i].sif_value = self._compute_sif(
                density=particles[i].density,
                connectivity=particles[i].connectivity,
                stability=particles[i].stability,
                energy=particles[i].energy,
                spatial_z=particles[i].spatial_z
            )
    
    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """
        计算两个张量的余弦相似度
        
        Args:
            a, b: 输入张量
        
        Returns:
            余弦相似度，范围[-1, 1]，归一化到[0, 1]
        """
        # 展平为1D
        a_flat = a.flatten()
        b_flat = b.flatten()
        
        # 计算余弦相似度
        sim = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1).item()
        
        # 归一化到[0, 1]
        sim = (sim + 1.0) / 2.0
        
        return sim


class InformationGroupBuilder:
    """
    信息组构建器 - 将粒子聚合成语义单元
    
    核心功能：
    1. 基于相似度聚类信息粒子
    2. 形成语义完整的信息组
    3. 纯规则方法（无神经网络）
    """
    
    def __init__(
        self, 
        similarity_threshold: float = 0.7,
        max_group_size: int = 10
    ):
        """
        Args:
            similarity_threshold: 相似度阈值（高于此值的粒子会被聚合）
            max_group_size: 最大组大小
        """
        self.similarity_threshold = similarity_threshold
        self.max_group_size = max_group_size
    
    def build_groups(
        self, 
        particles: List[InformationParticle]
    ) -> List[InformationGroup]:
        """
        构建信息组
        
        策略：
        - 基于连接度的贪心聚类
        - 相邻且相似的粒子聚合为一组
        
        Args:
            particles: 信息粒子列表
        
        Returns:
            信息组列表
        """
        if not particles:
            return []
        
        print(f"\n🔗 信息组构建开始...")
        print(f"   粒子数量: {len(particles)}")
        print(f"   相似度阈值: {self.similarity_threshold}")
        
        groups = []
        used = set()
        
        for i, particle in enumerate(particles):
            if i in used:
                continue
            
            # 创建新组
            group_particles = [particle]
            used.add(i)
            
            # 查找相邻的相似粒子
            for j in range(i + 1, min(i + self.max_group_size, len(particles))):
                if j in used:
                    continue
                
                # 检查与组中所有粒子的平均相似度
                avg_similarity = particles[j].connectivity
                
                if avg_similarity > self.similarity_threshold:
                    group_particles.append(particles[j])
                    used.add(j)
                else:
                    break  # 不再连续相似，停止
            
            # 创建信息组
            group = InformationGroup(
                particles=group_particles,
                group_id=len(groups)
            )
            groups.append(group)
        
        print(f"✅ 信息组构建完成，生成 {len(groups)} 个信息组")
        print(f"   平均组大小: {sum(g.group_size for g in groups) / len(groups):.2f}")
        print(f"   平均组SIF: {sum(g.group_sif for g in groups) / len(groups):.4f}")
        
        return groups


class LosslessReconstructor:
    """
    无损重构器 - 从信息粒子完美重构原始数据
    
    核心原理：
    - 直接提取和拼接粒子的raw_content
    - 按sequence_index排序保证顺序
    - 实现MSE=0的完美重构
    """
    
    def __init__(self):
        pass
    
    def reconstruct_from_particles(
        self, 
        particles: List[InformationParticle]
    ) -> torch.Tensor:
        """
        从信息粒子重构原始数据
        
        Args:
            particles: 信息粒子列表
        
        Returns:
            重构的数据，形状与原始输入相同
        """
        if not particles:
            return None
        
        print(f"\n🔄 无损重构开始...")
        print(f"   粒子数量: {len(particles)}")
        
        # 按sequence_index排序
        sorted_particles = sorted(particles, key=lambda p: p.sequence_index)
        
        # 拼接raw_content
        reconstructed_segments = [p.raw_content for p in sorted_particles]
        reconstructed = torch.cat(reconstructed_segments, dim=0)
        
        print(f"✅ 重构完成，输出形状: {reconstructed.shape}")
        
        return reconstructed
    
    def reconstruct_from_groups(
        self, 
        groups: List[InformationGroup]
    ) -> torch.Tensor:
        """
        从信息组重构原始数据
        
        Args:
            groups: 信息组列表
        
        Returns:
            重构的数据
        """
        if not groups:
            return None
        
        # 提取所有粒子
        all_particles = []
        for group in groups:
            all_particles.extend(group.particles)
        
        # 使用粒子重构
        return self.reconstruct_from_particles(all_particles)


class PureMathematicalSphereMapper:
    """
    纯数学的球面映射器 - 无神经网络
    
    核心功能：
    1. 将信息粒子映射到球面坐标(r, θ, φ)
    2. 将信息组映射到球面坐标
    3. 使用明确的数学公式（完全透明）
    
    数学公式：
    - r (径向): 反映抽象层次，基于稳定性和能量
    - θ (极角): 反映空间位置，基于spatial_x和spatial_y
    - φ (方位角): 反映时间和关联性，基于inner_time和connectivity
    """
    
    def __init__(self):
        pass
    
    def map_particle_to_sphere(
        self, 
        particle: InformationParticle
    ) -> Dict[str, Any]:
        """
        将信息粒子映射到球面坐标
        
        Args:
            particle: 信息粒子
        
        Returns:
            包含球面坐标和笛卡尔坐标的字典
        """
        # 获取球面坐标（使用粒子的内置方法）
        r, theta, phi = particle.get_sphere_coordinates()
        
        # 转换为笛卡尔坐标
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        return {
            'spherical': {'r': r, 'theta': theta, 'phi': phi},
            'cartesian': {'x': x, 'y': y, 'z': z},
            'particle_index': particle.sequence_index,
            'sif_value': particle.sif_value
        }
    
    def map_group_to_sphere(
        self, 
        group: InformationGroup
    ) -> Dict[str, Any]:
        """
        将信息组映射到球面坐标
        
        Args:
            group: 信息组
        
        Returns:
            包含球面坐标和笛卡尔坐标的字典
        """
        # 获取球面坐标（使用组的内置方法）
        r, theta, phi = group.get_sphere_coordinates()
        
        # 转换为笛卡尔坐标
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        return {
            'spherical': {'r': r, 'theta': theta, 'phi': phi},
            'cartesian': {'x': x, 'y': y, 'z': z},
            'group_id': group.group_id,
            'group_size': group.group_size,
            'group_sif': group.group_sif
        }
    
    def map_all_particles(
        self, 
        particles: List[InformationParticle]
    ) -> List[Dict[str, Any]]:
        """
        映射所有粒子到球面
        
        Args:
            particles: 粒子列表
        
        Returns:
            球面坐标列表
        """
        return [self.map_particle_to_sphere(p) for p in particles]
    
    def map_all_groups(
        self, 
        groups: List[InformationGroup]
    ) -> List[Dict[str, Any]]:
        """
        映射所有组到球面
        
        Args:
            groups: 组列表
        
        Returns:
            球面坐标列表
        """
        return [self.map_group_to_sphere(g) for g in groups]


# ============================================
# 测试和验证函数
# ============================================

def test_information_particle_system():
    """
    测试信息粒子系统的完整流程
    """
    print("="*70)
    print("  信息粒子系统测试")
    print("  Information Particle System Test")
    print("="*70)
    
    # 1. 创建测试数据
    print("\n📊 创建测试数据...")
    test_data = torch.randn(28, 28)  # 类似MNIST的28x28数据
    print(f"   测试数据形状: {test_data.shape}")
    
    # 2. 粒子化
    print("\n" + "="*70)
    extractor = InformationParticleExtractor(particle_size=28)
    particles = extractor.extract(test_data)
    
    # 3. 构建信息组
    print("\n" + "="*70)
    group_builder = InformationGroupBuilder(similarity_threshold=0.5)
    groups = group_builder.build_groups(particles)
    
    # 4. 球面映射
    print("\n" + "="*70)
    print("\n🌐 球面映射...")
    sphere_mapper = PureMathematicalSphereMapper()
    particle_coords = sphere_mapper.map_all_particles(particles)
    group_coords = sphere_mapper.map_all_groups(groups)
    
    print(f"✅ 映射完成")
    print(f"   粒子坐标数量: {len(particle_coords)}")
    print(f"   组坐标数量: {len(group_coords)}")
    
    # 打印第一个粒子的坐标示例
    if particle_coords:
        coord = particle_coords[0]
        print(f"\n   示例粒子坐标:")
        print(f"   - 球面: r={coord['spherical']['r']:.3f}, "
              f"θ={coord['spherical']['theta']:.3f}, "
              f"φ={coord['spherical']['phi']:.3f}")
        print(f"   - 笛卡尔: x={coord['cartesian']['x']:.3f}, "
              f"y={coord['cartesian']['y']:.3f}, "
              f"z={coord['cartesian']['z']:.3f}")
    
    # 5. 无损重构
    print("\n" + "="*70)
    reconstructor = LosslessReconstructor()
    reconstructed = reconstructor.reconstruct_from_particles(particles)
    
    # 6. 验证无损性
    print("\n" + "="*70)
    print("\n🔍 验证重构质量...")
    mse = F.mse_loss(reconstructed, test_data).item()
    cos_sim = F.cosine_similarity(
        reconstructed.flatten(), 
        test_data.flatten(), 
        dim=0
    ).item()
    
    print(f"✅ 重构验证:")
    print(f"   MSE: {mse:.10f}")
    print(f"   Cosine Similarity: {cos_sim:.10f}")
    
    if mse < 1e-6:
        print(f"   ✅ 完美重构！（MSE ≈ 0）")
    else:
        print(f"   ⚠️  重构有误差")
    
    print("\n" + "="*70)
    print("  测试完成！")
    print("="*70)
    
    return {
        'particles': particles,
        'groups': groups,
        'particle_coords': particle_coords,
        'group_coords': group_coords,
        'reconstructed': reconstructed,
        'mse': mse,
        'cosine_similarity': cos_sim
    }


if __name__ == '__main__':
    # 运行测试
    test_information_particle_system()

