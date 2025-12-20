import pandas as pd
import os

# -------------------------- 配置项（需根据实际路径修改） --------------------------
# 根目录：包含PicaBatch0~9文件夹的路径（绝对路径/相对路径均可）
root_dir = "results"  # 示例：如果文件夹在桌面，可改为 "C:/Users/你的用户名/Desktop/"
# 需要计算平均值的列（根据你的CSV列名调整）
columns_to_avg = ["total_time", "total_collisions", "avg_path_ratio"]

# -------------------------- 核心逻辑 --------------------------
# 初始化列表存储所有CSV的数据
all_data = []

# 遍历PicaBatch0到PicaBatch9
for batch_num in range(10):
    # 拼接文件夹和文件路径
    folder_name = f"PicaBatch{batch_num}"
    csv_path = os.path.join(root_dir, folder_name, "summary_results.csv")
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"⚠️  警告：文件不存在 - {csv_path}")
        continue
    
    # 读取CSV文件并添加到列表
    try:
        df = pd.read_csv(csv_path)
        all_data.append(df)
        print(f"✅ 成功读取：{csv_path}")
    except Exception as e:
        print(f"❌ 读取失败 - {csv_path}，错误：{e}")

# 合并所有数据
if not all_data:
    print("❌ 没有读取到任何有效数据！")
else:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 按scenario分组计算平均值
    avg_df = combined_df.groupby("scenario")[columns_to_avg].mean().reset_index()
    
    # 格式化输出（可选，让小数更易读）
    avg_df["avg_path_ratio"] = avg_df["avg_path_ratio"].round(8)  # 保留8位小数
    avg_df["total_time"] = avg_df["total_time"].round(1)  # 保留1位小数
    avg_df["total_collisions"] = avg_df["total_collisions"].round(1)  # 保留1位小数
    
    # -------------------------- 结果输出 --------------------------
    print("\n📊 各Scenario平均值结果：")
    print(avg_df)
    
    # 保存结果到新CSV文件（可选）
    output_path = os.path.join(root_dir, "average_summary_results.csv")
    avg_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n💾 结果已保存到：{output_path}")