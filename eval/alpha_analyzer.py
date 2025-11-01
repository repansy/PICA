import os
import numpy as np
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_headers(headers):
    """
    解析header并按后缀分类
    """
    classified_headers = defaultdict(list)
    header_indices = defaultdict(list)
    
    for idx, header in enumerate(headers):
        if header.endswith('_s'):
            classified_headers['_s'].append(header)
            header_indices['_s'].append(idx)
        elif header.endswith('_f'):
            classified_headers['_f'].append(header)
            header_indices['_f'].append(idx)
        elif header.endswith('_h'):
            classified_headers['_h'].append(header)
            header_indices['_h'].append(idx)
        elif header.endswith('_c'):
            classified_headers['_c'].append(header)
            header_indices['_c'].append(idx)
        else:
            classified_headers['other'].append(header)
            header_indices['other'].append(idx)
    
    return classified_headers, header_indices

# 假设headers是一个包含128个header的列表（32个块 × 4列）
# headers = [f"feature_{i}_s" for i in range(32)] + ...  # 根据实际情况调整 

def extract_nonzero_by_category(matrix_3d, header_indices, category):
    """
    提取指定类别的非零值
    """
    if category not in header_indices or not header_indices[category]:
        return None
    
    # 获取该类别的列索引（在4列中的局部索引）
    category_cols = header_indices[category]
    
    # 提取该类别的所有数据
    category_data = matrix_3d[:, category_cols, :]
    
    # 提取非零值
    nonzero_mask = category_data != 0
    nonzero_values = category_data[nonzero_mask]
    nonzero_positions = np.where(nonzero_mask)
    
    return {
        'nonzero_values': nonzero_values,
        'nonzero_positions': nonzero_positions,
        'nonzero_count': len(nonzero_values),
        'total_elements': category_data.size,
        'sparsity': 1 - len(nonzero_values) / category_data.size,
        'shape': category_data.shape,
        'category_data': category_data  # 保留原始数据用于后续分析
    }

def analyze_category_trends(category_result, category_name):
    """
    分析类别数据的变化趋势
    """
    if category_result is None:
        return None
    
    data_3d = category_result['category_data']
    nonzero_vals = category_result['nonzero_values']
    
    # 按深度维度分析非零值的变化
    depth_stats = []
    for depth in range(data_3d.shape[2]):
        depth_slice = data_3d[:, :, depth]
        nonzero_in_depth = depth_slice[depth_slice != 0]
        
        if len(nonzero_in_depth) > 0:
            depth_stats.append({
                'depth': depth,
                'nonzero_count': len(nonzero_in_depth),
                'nonzero_mean': np.mean(nonzero_in_depth),
                'nonzero_std': np.std(nonzero_in_depth),
                'nonzero_max': np.max(nonzero_in_depth),
                'nonzero_min': np.min(nonzero_in_depth)
            })
        else:
            depth_stats.append({
                'depth': depth,
                'nonzero_count': 0,
                'nonzero_mean': 0,
                'nonzero_std': 0,
                'nonzero_max': 0,
                'nonzero_min': 0
            })
    
    # 整体统计
    overall_stats = {
        'category': category_name,
        'total_nonzero': len(nonzero_vals),
        'sparsity': category_result['sparsity'],
        'nonzero_mean': np.mean(nonzero_vals) if len(nonzero_vals) > 0 else 0,
        'nonzero_std': np.std(nonzero_vals) if len(nonzero_vals) > 0 else 0,
        'depth_trend': depth_stats
    }
    
    return overall_stats

def analyze_nonzero_by_categories(matrix_3d, headers, column_idx):
    """
    主分析函数：按类别分析非零值变化
    """
    # 解析headers
    classified_headers, header_indices = parse_headers(headers)
    
    print(f"=== 纵列 {column_idx} 的Header分类结果 ===")
    for category, cat_headers in classified_headers.items():
        print(f"{category}: {len(cat_headers)}个 - {cat_headers[:3]}...")  # 显示前3个
    
    # 分析每个类别
    categories = ['_s', '_f', '_h', '_c', 'other']
    results = {}
    
    for category in categories:
        print(f"\n=== 分析 {category} 类别 ===")
        
        # 提取非零值
        category_result = extract_nonzero_by_category(matrix_3d, header_indices, category)
        
        if category_result is not None and category_result['nonzero_count'] > 0:
            # 分析趋势
            trend_analysis = analyze_category_trends(category_result, category)
            results[category] = trend_analysis
            
            print(f"非零值数量: {category_result['nonzero_count']}")
            print(f"稀疏度: {category_result['sparsity']:.4f}")
            print(f"非零值均值: {trend_analysis['nonzero_mean']:.4f}")
        else:
            print(f"类别 {category} 无非零值或不存在")
            results[category] = None
    
    return results, classified_headers, header_indices

def visualize_category_analysis(results, column_idx):
    """
    可视化各类别的非零值变化
    """
    if not any(results.values()):
        print("没有有效数据可可视化")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # 颜色映射
    colors = {'_s': 'red', '_f': 'blue', '_h': 'green', '_c': 'orange', 'other': 'purple'}
    
    # 1. 各类别非零值数量比较
    nonzero_counts = []
    category_labels = []
    sparsities = []
    
    for i, (category, result) in enumerate(results.items()):
        if result is not None:
            nonzero_counts.append(result['total_nonzero'])
            category_labels.append(category)
            sparsities.append(result['sparsity'])
    
    if nonzero_counts:
        axes[0].bar(category_labels, nonzero_counts, 
                   color=[colors.get(cat, 'gray') for cat in category_labels])
        axes[0].set_title(f'纵列 {column_idx} - 各类别非零值数量')
        axes[0].set_ylabel('非零值数量')
        
        # 在柱子上显示数值
        for i, v in enumerate(nonzero_counts):
            axes[0].text(i, v, str(v), ha='center', va='bottom')
    
    # 2. 稀疏度比较
    if sparsities:
        axes[1].bar(category_labels, sparsities,
                   color=[colors.get(cat, 'gray') for cat in category_labels])
        axes[1].set_title(f'纵列 {column_idx} - 各类别稀疏度')
        axes[1].set_ylabel('稀疏度')
        axes[1].set_ylim(0, 1)
    
    # 3. 各类别随时间的变化趋势（非零值数量）
    axes[2].set_title(f'纵列 {column_idx} - 非零值数量随时间变化')
    axes[2].set_xlabel('时间/深度')
    axes[2].set_ylabel('非零值数量')
    
    # 4. 各类别随时间的变化趋势（非零值均值）
    axes[3].set_title(f'纵列 {column_idx} - 非零值均值随时间变化')
    axes[3].set_xlabel('时间/深度')
    axes[3].set_ylabel('非零值均值')
    
    for category, result in results.items():
        if result is not None and len(result['depth_trend']) > 0:
            depths = [stat['depth'] for stat in result['depth_trend']]
            nonzero_counts = [stat['nonzero_count'] for stat in result['depth_trend']]
            nonzero_means = [stat['nonzero_mean'] for stat in result['depth_trend']]
            
            axes[2].plot(depths, nonzero_counts, 
                        label=category, color=colors.get(category, 'gray'), marker='o', markersize=2)
            axes[3].plot(depths, nonzero_means, 
                        label=category, color=colors.get(category, 'gray'), marker='s', markersize=2)
    
    for ax in axes[2:]:
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 额外：非零值分布直方图
    plt.figure(figsize=(12, 8))
    for i, (category, result) in enumerate(results.items()):
        if result is not None and result['total_nonzero'] > 0:
            # 提取所有非零值
            all_nonzero = []
            for stat in result['depth_trend']:
                if stat['nonzero_count'] > 0:
                    # 这里需要从原始数据中提取，简化显示用统计值
                    all_nonzero.extend([stat['nonzero_mean']] * stat['nonzero_count'])
            
            if all_nonzero:
                plt.hist(all_nonzero, bins=30, alpha=0.6, 
                        label=f'{category} (n={result["total_nonzero"]})', 
                        color=colors.get(category, 'gray'))
    
    plt.title(f'纵列 {column_idx} - 各类别非零值分布')
    plt.xlabel('非零值大小')
    plt.ylabel('频数')
    plt.legend()
    plt.show()

    # 完整的使用流程
def complete_analysis(original_matrix, headers, column_idx, t=240):
    """
    完整的分析流程
    """
    # 1. 提取指定纵列的三维矩阵
    matrix_3d = extract_column_to_3d(original_matrix, column_idx, t)
    print(f"提取的纵列 {column_idx} 三维矩阵形状: {matrix_3d.shape}")
    
    # 2. 按类别分析非零值
    results, classified_headers, header_indices = analyze_nonzero_by_categories(
        matrix_3d, headers, column_idx)
    
    # 3. 可视化分析结果
    visualize_category_analysis(results, column_idx)
    
    # 4. 输出详细统计
    print(f"\n=== 纵列 {column_idx} 详细统计 ===")
    for category, result in results.items():
        if result is not None:
            print(f"\n{category} 类别:")
            print(f"  总非零值: {result['total_nonzero']}")
            print(f"  稀疏度: {result['sparsity']:.4f}")
            print(f"  非零值均值: {result['nonzero_mean']:.4f}")
            print(f"  非零值标准差: {result['nonzero_std']:.4f}")
    
    return results, matrix_3d

# 使用示例
# 假设有headers列表和原始矩阵
# results, matrix_3d = complete_analysis(original_matrix, headers, column_idx=0)

def analyze_multiple_columns(original_matrix, headers, column_indices, t=240):
    """
    批量分析多个纵列
    """
    all_results = {}
    
    for col_idx in column_indices:
        print(f"\n{'='*50}")
        print(f"分析纵列 {col_idx}")
        print(f"{'='*50}")
        
        results, matrix_3d = complete_analysis(original_matrix, headers, col_idx, t)
        all_results[col_idx] = {
            'results': results,
            'matrix_3d': matrix_3d
        }
    
    # 跨纵列比较分析
    compare_across_columns(all_results)
    
    return all_results

def compare_across_columns(all_results):
    """
    跨纵列比较分析
    """
    categories = ['_s', '_f', '_h', '_c']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, category in enumerate(categories):
        if i >= len(axes):
            break
            
        col_indices = []
        nonzero_counts = []
        sparsities = []
        
        for col_idx, result_data in all_results.items():
            results = result_data['results']
            if category in results and results[category] is not None:
                col_indices.append(col_idx)
                nonzero_counts.append(results[category]['total_nonzero'])
                sparsities.append(results[category]['sparsity'])
        
        if col_indices:
            axes[i].bar(col_indices, nonzero_counts, alpha=0.7)
            axes[i].set_title(f'{category} 类别跨纵列比较')
            axes[i].set_xlabel('纵列索引')
            axes[i].set_ylabel('非零值数量')
    
    plt.tight_layout()
    plt.show()

# 批量分析示例
# all_results = analyze_multiple_columns(original_matrix, headers, [0, 1, 2, 3])