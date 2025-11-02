"""
Baseline对比实验

对比方法：
1. 直接存储（Identity Mapping）
2. PCA降维
3. 我们的信息粒子系统

作者：北京求一数生科技中心
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from information_oriented_system_v2 import InformationOrientedSystemV2


class DirectStorageBaseline:
    """Baseline 1: 直接存储（最简单的baseline）"""
    
    def __init__(self):
        self.name = "Direct Storage"
        self.stored_data = None
    
    def process(self, data):
        """直接存储"""
        start = time.time()
        self.stored_data = data.clone()
        process_time = (time.time() - start) * 1000
        
        return {
            'process_time': process_time,
            'compressed_size': data.numel(),  # 没有压缩
            'storage': self.stored_data
        }
    
    def reconstruct(self):
        """直接返回"""
        start = time.time()
        reconstructed = self.stored_data
        recon_time = (time.time() - start) * 1000
        
        return reconstructed, recon_time


class PCABaseline:
    """Baseline 2: PCA降维"""
    
    def __init__(self, n_components=64, fit_data=None):
        self.name = f"PCA (n={n_components})"
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.fitted = False
        self.compressed = None
        self.mean = None
        self.original_shape = None
        
        # 预先fit
        if fit_data is not None:
            self.pca.fit(fit_data)
            self.fitted = True
    
    def process(self, data):
        """PCA降维"""
        start = time.time()
        
        # 展平
        self.original_shape = data.shape
        data_flat = data.flatten().cpu().numpy().reshape(1, -1)
        
        # PCA
        self.compressed = self.pca.transform(data_flat)
        
        process_time = (time.time() - start) * 1000
        
        return {
            'process_time': process_time,
            'compressed_size': self.compressed.size,
            'storage': self.compressed
        }
    
    def reconstruct(self):
        """PCA重构"""
        start = time.time()
        
        reconstructed_flat = self.pca.inverse_transform(self.compressed)
        reconstructed = torch.tensor(reconstructed_flat, dtype=torch.float32)
        reconstructed = reconstructed.reshape(self.original_shape)
        
        recon_time = (time.time() - start) * 1000
        
        return reconstructed, recon_time


class OurMethod:
    """我们的信息粒子系统"""
    
    def __init__(self, particle_size=4):
        self.name = f"Information Particle System (psize={particle_size})"
        self.system = InformationOrientedSystemV2(particle_size=particle_size)
        self.output = None
    
    def process(self, data):
        """信息粒子化"""
        start = time.time()
        self.output = self.system.forward(data)
        process_time = (time.time() - start) * 1000
        
        # 计算存储大小（粒子数 × 12维特征 + raw_content）
        num_particles = self.output['num_particles']
        compressed_size = num_particles * 12  # 12维特征
        # raw_content仍然需要存储，但有结构化信息
        
        return {
            'process_time': process_time,
            'compressed_size': compressed_size,
            'num_particles': num_particles,
            'num_groups': self.output['num_groups'],
            'avg_sif': self.output['avg_sif'],
            'storage': self.output
        }
    
    def reconstruct(self):
        """无损重构"""
        start = time.time()
        reconstructed = self.system.reconstruct(self.output)
        recon_time = (time.time() - start) * 1000
        
        return reconstructed, recon_time


def compare_methods(image, label, methods):
    """对比所有方法"""
    
    results = {}
    
    for method in methods:
        # 处理
        process_result = method.process(image)
        
        # 重构
        reconstructed, recon_time = method.reconstruct()
        
        # 验证
        mse = F.mse_loss(reconstructed, image).item()
        cos_sim = F.cosine_similarity(
            reconstructed.flatten(),
            image.flatten(),
            dim=0
        ).item()
        
        results[method.name] = {
            'process_time': process_result['process_time'],
            'recon_time': recon_time,
            'total_time': process_result['process_time'] + recon_time,
            'mse': mse,
            'cosine_sim': cos_sim,
            'compressed_size': process_result.get('compressed_size', 0),
            'num_particles': process_result.get('num_particles', 'N/A'),
            'num_groups': process_result.get('num_groups', 'N/A'),
            'avg_sif': process_result.get('avg_sif', 'N/A'),
            'perfect': mse < 1e-6
        }
    
    return results


def run_comparison(num_samples=50, dataset='mnist'):
    """运行完整对比实验"""
    
    print("="*70)
    print(f"  Baseline对比实验 - {dataset.upper()}")
    print("="*70)
    
    # 加载数据
    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset == 'mnist':
        testset = torchvision.datasets.MNIST(
            root='./data/MNIST', train=False, download=False, transform=transform
        )
    elif dataset == 'fashion':
        testset = torchvision.datasets.FashionMNIST(
            root='./data/FashionMNIST', train=False, download=False, transform=transform
        )
    
    # 准备PCA训练数据（用前200个样本）
    print("\n准备PCA训练数据...")
    pca_train_data = []
    for i in range(min(200, len(testset))):
        img, _ = testset[i]
        pca_train_data.append(img.squeeze(0).flatten().numpy())
    pca_train_data = np.array(pca_train_data)
    print(f"  PCA训练数据形状: {pca_train_data.shape}")
    
    # 创建方法
    methods = [
        DirectStorageBaseline(),
        PCABaseline(n_components=64, fit_data=pca_train_data),
        PCABaseline(n_components=128, fit_data=pca_train_data),
        OurMethod(particle_size=4)
    ]
    
    print(f"\n对比方法:")
    for i, m in enumerate(methods):
        print(f"  {i+1}. {m.name}")
    
    # 测试
    print(f"\n测试 {num_samples} 个样本...")
    
    all_results = {m.name: [] for m in methods}
    
    for i in range(num_samples):
        image, label = testset[i]
        image_2d = image.squeeze(0)
        
        results = compare_methods(image_2d, label, methods)
        
        for method_name, result in results.items():
            all_results[method_name].append(result)
        
        if (i+1) % 10 == 0:
            print(f"  处理: {i+1}/{num_samples}")
    
    # 统计
    print("\n" + "="*70)
    print("📊 结果汇总")
    print("="*70)
    
    print(f"\n{'方法':<35} {'处理(ms)':<12} {'重构(ms)':<12} {'MSE':<15} {'完美率':<10}")
    print("-"*70)
    
    for method_name in all_results:
        results = all_results[method_name]
        
        avg_process = np.mean([r['process_time'] for r in results])
        avg_recon = np.mean([r['recon_time'] for r in results])
        avg_mse = np.mean([r['mse'] for r in results])
        perfect_rate = sum([r['perfect'] for r in results]) / len(results)
        
        print(f"{method_name:<35} {avg_process:<12.2f} {avg_recon:<12.2f} "
              f"{avg_mse:<15.10f} {perfect_rate*100:<10.1f}%")
    
    # 详细对比
    print(f"\n" + "="*70)
    print("🔍 详细对比")
    print("="*70)
    
    # 找出我们的方法
    our_results = all_results[[k for k in all_results if 'Information' in k][0]]
    direct_results = all_results['Direct Storage']
    pca64_results = all_results['PCA (n=64)']
    
    print(f"\n1. 重构质量对比:")
    print(f"   Direct Storage:  MSE = {np.mean([r['mse'] for r in direct_results]):.10f}")
    print(f"   PCA (n=64):      MSE = {np.mean([r['mse'] for r in pca64_results]):.10f}")
    print(f"   Ours:            MSE = {np.mean([r['mse'] for r in our_results]):.10f}")
    
    print(f"\n2. 额外信息:")
    if our_results[0]['num_particles'] != 'N/A':
        print(f"   平均粒子数:  {np.mean([r['num_particles'] for r in our_results]):.1f}")
        print(f"   平均信息组:  {np.mean([r['num_groups'] for r in our_results]):.1f}")
        print(f"   平均SIF值:   {np.mean([r['avg_sif'] for r in our_results]):.4f}")
        print(f"   ✅ 我们的方法提供了额外的结构化信息！")
    
    print(f"\n3. 关键优势:")
    our_mse = np.mean([r['mse'] for r in our_results])
    pca_mse = np.mean([r['mse'] for r in pca64_results])
    
    if our_mse < 1e-6:
        print(f"   ✅ 我们实现完美重构（MSE≈0）")
        print(f"   ✅ PCA有信息损失（MSE={pca_mse:.6f}）")
    
    print(f"   ✅ 我们提供12维可解释特征")
    print(f"   ✅ 我们提供信息组结构")
    print(f"   ✅ 我们提供SIF质量评估")
    
    return all_results


def visualize_comparison(all_results, save_path='baseline_comparison.png'):
    """可视化对比结果"""
    
    print(f"\n🎨 生成对比图表...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    methods = list(all_results.keys())
    colors = ['gray', 'blue', 'cyan', 'red']
    
    # 1. MSE对比
    mse_values = [np.mean([r['mse'] for r in all_results[m]]) for m in methods]
    axes[0].bar(range(len(methods)), mse_values, color=colors)
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=8)
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Reconstruction Error (MSE)')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 处理时间对比
    time_values = [np.mean([r['total_time'] for r in all_results[m]]) for m in methods]
    axes[1].bar(range(len(methods)), time_values, color=colors)
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=8)
    axes[1].set_ylabel('Time (ms)')
    axes[1].set_title('Total Processing Time')
    axes[1].grid(True, alpha=0.3)
    
    # 3. 完美重构率
    perfect_rates = [sum([r['perfect'] for r in all_results[m]])/len(all_results[m])*100 
                     for m in methods]
    axes[2].bar(range(len(methods)), perfect_rates, color=colors)
    axes[2].set_xticks(range(len(methods)))
    axes[2].set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=8)
    axes[2].set_ylabel('Perfect Reconstruction Rate (%)')
    axes[2].set_title('Perfect Reconstruction Rate')
    axes[2].set_ylim([0, 105])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ 保存到: {save_path}")
    plt.close()


if __name__ == '__main__':
    # MNIST测试
    mnist_results = run_comparison(num_samples=50, dataset='mnist')
    visualize_comparison(mnist_results, 'baseline_comparison_mnist.png')
    
    print("\n" + "="*70)
    
    # Fashion-MNIST测试
    fashion_results = run_comparison(num_samples=50, dataset='fashion')
    visualize_comparison(fashion_results, 'baseline_comparison_fashion.png')
    
    print("\n" + "="*70)
    print("  对比实验完成！")
    print("="*70)

