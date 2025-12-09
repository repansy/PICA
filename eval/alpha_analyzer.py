import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_nonzero_nums_and_plot_histogram(csv_path, bins=30, save_fig_path=None, fig_size=(10, 6)):
    """
    从CSV文件中提取所有数字的非零值，并绘制直方图
    
    参数:
    csv_path: str - CSV文件路径
    bins: int - 直方图的分箱数量（默认30）
    save_fig_path: str - 图片保存路径（None表示不保存，默认None）
    fig_size: tuple - 图片尺寸（默认(10,6)）
    """
    # 1. 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取CSV文件，数据形状：{df.shape}")
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_path}")
        return
    except Exception as e:
        print(f"读取文件时出错：{str(e)}")
        return
    
    # 2. 筛选数字类型的列（int, float）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("警告：CSV文件中没有数字类型的列")
        return
    print(f"数字类型列：{numeric_cols}")
    
    # 3. 收集所有非零的数字值
    nonzero_values = []
    for col in numeric_cols:
        # 提取该列非空且非零的值
        col_values = df[col].dropna()  # 去除缺失值
        nonzero_col = col_values[col_values != 0]  # 筛选非零值
        nonzero_values.extend(nonzero_col.tolist())
    
    # 4. 处理无数据的情况
    if not nonzero_values:
        print("警告：所有数字列中没有非零值")
        return
    
    # 转换为numpy数组（便于后续处理）
    nonzero_values = np.array(nonzero_values)
    print(f"共提取到 {len(nonzero_values)} 个非零数字")
    print(f"数值范围：[{nonzero_values.min():.4f}, {nonzero_values.max():.4f}]")
    print(f"数值均值：{nonzero_values.mean():.4f}")
    print(f"数值标准差：{nonzero_values.std():.4f}")
    
    # 5. 绘制直方图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文（Windows）
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统使用这行
    plt.rcParams['axes.unicode_minus'] = False  # 支持负号
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    # 绘制直方图（edgecolor使柱子更清晰）
    n, bins_edges, patches = ax.hist(
        nonzero_values, 
        bins=bins, 
        edgecolor='black', 
        alpha=0.7, 
        color='#1f77b4'
    )
    
    # 设置图表样式
    ax.set_title('CSV文件中所有非零数字的直方图', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('数值', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    
    # 添加网格线（提高可读性）
    ax.grid(True, alpha=0.3, axis='y')
    
    # 调整布局（防止标签被截断）
    plt.tight_layout()
    
    # 保存图片（如果指定路径）
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        print(f"直方图已保存到：{save_fig_path}")
    
    # 显示图片
    plt.show()
    
    # 返回统计信息（可选）
    return {
        '总非零值数量': len(nonzero_values),
        '最小值': nonzero_values.min(),
        '最大值': nonzero_values.max(),
        '均值': nonzero_values.mean(),
        '标准差': nonzero_values.std(),
        '分箱数量': bins,
        '频数统计': n.tolist(),
        '分箱边界': bins_edges.tolist()
    }

# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 请替换为你的CSV文件路径
    csv_file_path = "F:\\CodeRepo\\PICA\\results\\PicaBatch\\spheretest\\RA1_37_alpha.csv"
    
    # 调用函数（可根据需要调整参数）
    result = extract_nonzero_nums_and_plot_histogram(
        csv_path=csv_file_path,
        bins=50,  # 分箱数量（根据数据分布调整）
        save_fig_path="nonzero_histogram.png",  # 保存图片（可选）
        fig_size=(12, 7)
    )
    
    # 打印统计结果（如果有数据）
    if result:
        print("\n=== 统计结果 ===")
        for key, value in result.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")