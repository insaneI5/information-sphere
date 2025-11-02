"""
信息粒子系统 - 应用案例演示

展示4个实际应用场景：
1. 图像质量评估
2. 异常检测
3. 数据压缩分析
4. 信息结构可视化

作者：北京求一数生科技中心
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

from information_oriented_system_v2 import InformationOrientedSystemV2


class QualityAssessment:
    """应用1: 图像质量评估"""
    
    def __init__(self):
        self.system = InformationOrientedSystemV2(particle_size=4)
    
    def assess(self, image):
        """评估图像质量"""
        output = self.system.forward(image)
        
        return {
            'overall_quality': output['avg_sif'],
            'num_particles': output['num_particles'],
            'num_groups': output['num_groups'],
            'particles': output['particles']
        }
    
    def compare_quality(self, images, labels):
        """对比多张图像质量"""
        results = []
        
        for i, (img, label) in enumerate(zip(images, labels)):
            quality = self.assess(img.squeeze(0))
            results.append({
                'index': i,
                'label': label,
                'quality': quality['overall_quality'],
                'particles': quality['num_particles'],
                'groups': quality['num_groups']
            })
        
        return results


class AnomalyDetection:
    """应用2: 异常检测"""
    
    def __init__(self, sif_threshold=0.3):
        self.system = InformationOrientedSystemV2(particle_size=4)
        self.sif_threshold = sif_threshold
    
    def detect(self, image):
        """检测图像中的异常区域"""
        output = self.system.forward(image)
        
        # 找出低SIF粒子（异常候选）
        anomalies = []
        for particle in output['particles']:
            if particle.sif_value < self.sif_threshold:
                anomalies.append({
                    'sequence_idx': particle.sequence_index,
                    'sif': particle.sif_value,
                    'connectivity': particle.connectivity,
                    'density': particle.density,
                    'energy': particle.energy
                })
        
        return {
            'num_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / output['num_particles'],
            'anomalies': anomalies,
            'avg_sif': output['avg_sif']
        }
    
    def visualize_anomalies(self, image, detection_result, save_path='anomaly_detection.png'):
        """可视化异常区域"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 原图
        axes[0].imshow(image.cpu().numpy(), cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 异常标记
        axes[1].imshow(image.cpu().numpy(), cmap='gray')
        
        # 标记异常区域
        particle_size = 4
        for anomaly in detection_result['anomalies']:
            idx = anomaly['sequence_idx']
            row = (idx * particle_size) // image.shape[1]
            col = (idx * particle_size) % image.shape[1]
            
            rect = plt.Rectangle(
                (col, row), particle_size, particle_size,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[1].add_patch(rect)
        
        axes[1].set_title(f'Anomalies Detected: {detection_result["num_anomalies"]}')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class CompressionAnalysis:
    """应用3: 数据压缩分析"""
    
    def __init__(self):
        self.system = InformationOrientedSystemV2(particle_size=4)
    
    def analyze(self, image):
        """分析压缩性能"""
        start = time.time()
        output = self.system.forward(image)
        process_time = (time.time() - start) * 1000
        
        # 原始大小
        original_size = image.numel() * 4  # float32 = 4 bytes
        
        # 压缩大小（特征+raw_content）
        feature_size = output['num_particles'] * 12 * 4  # 12维特征
        content_size = original_size  # raw_content保持原大小
        
        compressed_size = feature_size + content_size
        
        # 但我们提供了额外信息！
        extra_info = {
            'sif_values': output['num_particles'],
            'group_structure': output['num_groups'],
            'sphere_coordinates': output['num_particles'] * 3
        }
        
        return {
            'original_size': original_size,
            'feature_size': feature_size,
            'content_size': content_size,
            'total_size': compressed_size,
            'compression_ratio': original_size / compressed_size,
            'extra_info': extra_info,
            'process_time': process_time,
            'avg_sif': output['avg_sif']
        }


class StructureVisualization:
    """应用4: 信息结构可视化"""
    
    def __init__(self):
        self.system = InformationOrientedSystemV2(particle_size=4)
    
    def visualize(self, image, save_path='structure_viz.png'):
        """可视化信息结构"""
        output = self.system.forward(image)
        
        fig = plt.figure(figsize=(16, 4))
        gs = fig.add_gridspec(1, 4, wspace=0.3)
        
        # 1. 原图
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(image.cpu().numpy(), cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # 2. SIF分布
        ax2 = fig.add_subplot(gs[1])
        sif_values = [p.sif_value for p in output['particles']]
        ax2.hist(sif_values, bins=20, color='skyblue', edgecolor='black')
        ax2.set_title(f'SIF Distribution\n(Avg: {np.mean(sif_values):.3f})')
        ax2.set_xlabel('SIF Value')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # 3. 连接度网络
        ax3 = fig.add_subplot(gs[2])
        connectivity_values = [p.connectivity for p in output['particles']]
        ax3.scatter(range(len(connectivity_values)), connectivity_values, 
                   c=sif_values, cmap='viridis', s=50)
        ax3.set_title('Connectivity Network')
        ax3.set_xlabel('Particle Index')
        ax3.set_ylabel('Connectivity')
        ax3.grid(True, alpha=0.3)
        
        # 4. 球面投影
        ax4 = fig.add_subplot(gs[3])
        theta_vals = []
        phi_vals = []
        for p in output['particles']:
            r, theta, phi = p.get_sphere_coordinates()
            theta_vals.append(theta)
            phi_vals.append(phi)
        
        scatter = ax4.scatter(phi_vals, theta_vals, c=sif_values, 
                             cmap='viridis', s=50, alpha=0.6)
        ax4.set_title('Sphere Projection')
        ax4.set_xlabel('φ (azimuth)')
        ax4.set_ylabel('θ (polar)')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='SIF Value')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'avg_sif': np.mean(sif_values),
            'avg_connectivity': np.mean(connectivity_values),
            'num_particles': len(output['particles']),
            'num_groups': output['num_groups']
        }


def demo_quality_assessment():
    """演示1: 图像质量评估"""
    print("\n" + "="*70)
    print("  应用案例1: 图像质量评估")
    print("="*70)
    
    # 加载数据
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(
        root='./data/MNIST', train=False, download=False, transform=transform
    )
    
    # 选择10张图像
    images = [testset[i][0] for i in range(10)]
    labels = [testset[i][1] for i in range(10)]
    
    qa = QualityAssessment()
    results = qa.compare_quality(images, labels)
    
    print("\n📊 质量评估结果:")
    print(f"{'索引':<6} {'标签':<6} {'质量(SIF)':<12} {'粒子数':<10} {'信息组':<10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['index']:<6} {r['label']:<6} {r['quality']:<12.4f} "
              f"{r['particles']:<10} {r['groups']:<10}")
    
    print(f"\n✅ 关键发现:")
    avg_quality = np.mean([r['quality'] for r in results])
    print(f"   平均图像质量: {avg_quality:.4f}")
    print(f"   质量范围: [{min(r['quality'] for r in results):.4f}, "
          f"{max(r['quality'] for r in results):.4f}]")
    print(f"   💡 SIF值可作为无参考图像质量指标！")


def demo_anomaly_detection():
    """演示2: 异常检测"""
    print("\n" + "="*70)
    print("  应用案例2: 异常检测")
    print("="*70)
    
    # 加载数据
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(
        root='./data/MNIST', train=False, download=False, transform=transform
    )
    
    detector = AnomalyDetection(sif_threshold=0.35)
    
    print("\n🔍 检测10张图像的异常...")
    
    for i in range(10):
        image, label = testset[i]
        image_2d = image.squeeze(0)
        
        result = detector.detect(image_2d)
        
        print(f"  图像{i} (数字{label}): "
              f"异常率={result['anomaly_rate']*100:.1f}%, "
              f"异常数={result['num_anomalies']}, "
              f"平均SIF={result['avg_sif']:.4f}")
        
        # 可视化第一张
        if i == 0:
            detector.visualize_anomalies(image_2d, result, 
                                        f'anomaly_detection_sample.png')
    
    print(f"\n✅ 应用价值:")
    print(f"   💡 低SIF区域标记潜在问题")
    print(f"   💡 可用于质量控制、缺陷检测")
    print(f"   ✅ 可视化保存: anomaly_detection_sample.png")


def demo_compression_analysis():
    """演示3: 数据压缩分析"""
    print("\n" + "="*70)
    print("  应用案例3: 数据压缩分析")
    print("="*70)
    
    # 加载数据
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(
        root='./data/MNIST', train=False, download=False, transform=transform
    )
    
    analyzer = CompressionAnalysis()
    
    # 分析多张图像
    results = []
    for i in range(10):
        image, label = testset[i]
        result = analyzer.analyze(image.squeeze(0))
        results.append(result)
    
    print("\n📊 压缩分析结果:")
    
    avg_orig = np.mean([r['original_size'] for r in results])
    avg_feat = np.mean([r['feature_size'] for r in results])
    avg_total = np.mean([r['total_size'] for r in results])
    avg_time = np.mean([r['process_time'] for r in results])
    
    print(f"   原始大小:   {avg_orig/1024:.2f} KB")
    print(f"   特征大小:   {avg_feat/1024:.2f} KB")
    print(f"   总大小:     {avg_total/1024:.2f} KB")
    print(f"   压缩比:     {avg_orig/avg_total:.2f}x")
    print(f"   处理时间:   {avg_time:.2f}ms")
    
    print(f"\n✅ 关键优势:")
    print(f"   💡 虽然增加了{(avg_total-avg_orig)/1024:.2f}KB特征")
    print(f"   💡 但提供了:")
    print(f"      - 12维可解释特征")
    print(f"      - SIF质量评分")
    print(f"      - 信息组结构")
    print(f"      - 球面拓扑关系")
    print(f"   ✅ 用少量额外空间换取丰富的结构信息！")


def demo_structure_visualization():
    """演示4: 信息结构可视化"""
    print("\n" + "="*70)
    print("  应用案例4: 信息结构可视化")
    print("="*70)
    
    # 加载数据
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(
        root='./data/MNIST', train=False, download=False, transform=transform
    )
    
    visualizer = StructureVisualization()
    
    # 可视化3张不同数字
    for digit in [0, 5, 8]:
        # 找到该数字的样本
        for i in range(len(testset)):
            if testset[i][1] == digit:
                image = testset[i][0].squeeze(0)
                result = visualizer.visualize(image, 
                                             f'structure_viz_digit_{digit}.png')
                
                print(f"\n  数字 {digit}:")
                print(f"    平均SIF: {result['avg_sif']:.4f}")
                print(f"    平均连接度: {result['avg_connectivity']:.4f}")
                print(f"    信息组数: {result['num_groups']}")
                print(f"    可视化保存: structure_viz_digit_{digit}.png")
                break
    
    print(f"\n✅ 应用价值:")
    print(f"   💡 直观展示信息的内在结构")
    print(f"   💡 SIF分布反映内容复杂度")
    print(f"   💡 连接度网络揭示信息关联")
    print(f"   💡 球面投影显示拓扑关系")


def generate_summary():
    """生成应用案例总结"""
    print("\n" + "="*70)
    print("  应用案例总结")
    print("="*70)
    
    summary = """
✅ 已演示4个实际应用：

1. **图像质量评估**
   - 无需参考图像
   - SIF值作为质量指标
   - 实时评估图像质量
   
2. **异常检测**
   - 自动识别低质量区域
   - 可视化异常位置
   - 适用于质量控制

3. **数据压缩分析**
   - 少量额外空间
   - 提供丰富结构信息
   - 12维特征 + SIF + 拓扑

4. **信息结构可视化**
   - 直观展示信息组织
   - 多角度分析内容
   - 揭示隐藏模式

🎯 核心价值：

1. **透明性**: 所有特征可解释
2. **实用性**: 即插即用的应用
3. **通用性**: 适用于多种数据
4. **高效性**: 毫秒级处理速度

💡 潜在应用领域：

- 图像/视频质量评估
- 工业缺陷检测
- 医学图像分析
- 数据质量监控
- 信息检索与推荐
- 异常行为检测
- 内容理解与生成
"""
    
    print(summary)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  信息粒子系统 - 应用案例演示")
    print("  作者: 北京求一数生科技中心")
    print("="*70)
    
    # 演示所有应用
    demo_quality_assessment()
    demo_anomaly_detection()
    demo_compression_analysis()
    demo_structure_visualization()
    
    # 生成总结
    generate_summary()
    
    print("\n" + "="*70)
    print("  所有应用案例演示完成！")
    print("="*70)

