import pandas as pd
import numpy as np
import os
import re
from natsort import natsorted

# ==================== 配置 ====================
INPUT_FOLDER  = 'data'                     # 原始图片文件夹
INPUT_EXCEL   = 'result.xlsx'              # 原始推理结果（必须包含：实验编号, X坐标, Y坐标, 置信度）
OUTPUT_EXCEL  = 'result_final.xlsx'        # 最终结果
DISTANCE_THRESHOLD = 1000                    # 相邻有效帧的欧氏距离阈值，大于此值则开启新实验
# ==============================================

def generate_all_ids():
    """生成所有合法实验编号（a0b0c），a=1..5, b=1..5，c 一般为 1~3，但 1010 和 1020 的 c 为 1~4"""
    ids = []
    for a in range(1, 6):
        for b in range(1, 6):
            # 只有 a=1,b=1 和 a=1,b=2 时 c 取到 4
            if (a == 1 and b == 1) or (a == 1 and b == 2):
                c_max = 4
            else:
                c_max = 3
            for c in range(1, c_max + 1):
                ids.append(f"{a}0{b}0{c}")
    return ids

def read_and_align_data(excel_path, image_folder):
    """读取 Excel，并从图片文件夹获取排序后的图片名，将表格行与图片名一一对应"""
    df = pd.read_excel(excel_path)
    print(f"原始表格行数：{len(df)}")

    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"图片文件夹 {image_folder} 不存在！")
    img_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    img_files = natsorted(img_files)
    print(f"图片文件夹中共有 {len(img_files)} 张图片")

    min_len = min(len(df), len(img_files))
    if len(df) != len(img_files):
        print(f"⚠️ 警告：表格行数({len(df)})与图片数量({len(img_files)})不一致！将截取前 {min_len} 行处理。")
        df = df.iloc[:min_len].copy()
        img_files = img_files[:min_len]
    else:
        df = df.copy()

    df['图片名称'] = img_files
    return df

def is_valid(row):
    try:
        float(row['X坐标'])
        float(row['Y坐标'])
        return True
    except (ValueError, TypeError):
        return False

def distance(row1, row2):
    x1, y1 = float(row1['X坐标']), float(row1['Y坐标'])
    x2, y2 = float(row2['X坐标']), float(row2['Y坐标'])
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def cluster_frames(df, threshold):
    groups = []
    current_group = []
    last_valid_idx = None

    for idx, row in df.iterrows():
        if is_valid(row):
            if last_valid_idx is None:
                current_group = [idx]
            else:
                dist = distance(df.loc[last_valid_idx], row)
                if dist > threshold:
                    groups.append(current_group)
                    current_group = [idx]
                else:
                    current_group.append(idx)
            last_valid_idx = idx

    if current_group:
        groups.append(current_group)
    return groups

def main():
    full_ids = generate_all_ids()
    print(f"理论实验编号总数：{len(full_ids)}")

    df = read_and_align_data(INPUT_EXCEL, INPUT_FOLDER)

    groups = cluster_frames(df, DISTANCE_THRESHOLD)
    actual_groups = len(groups)
    print(f"根据阈值 {DISTANCE_THRESHOLD}，实际聚出实验组数：{actual_groups}")

    if actual_groups != len(full_ids):
        print(f"⚠️ 警告：实际组数({actual_groups})与理论编号数({len(full_ids)})不匹配！请调整 DISTANCE_THRESHOLD。")
        if actual_groups > len(full_ids):
            extra = [f"extra_{i}" for i in range(1, actual_groups - len(full_ids) + 1)]
            assign_ids = full_ids + extra
        else:
            assign_ids = full_ids[:actual_groups]
    else:
        assign_ids = full_ids

    result_rows = []
    for g_idx, group_indices in enumerate(groups):
        exp_id = assign_ids[g_idx]
        for idx in group_indices:
            row = df.loc[idx]
            result_rows.append({
                '实验编号': exp_id,
                'X坐标': row['X坐标'],
                'Y坐标': row['Y坐标'],
                '置信度': row.get('置信度', ''),
                '图片名称': row['图片名称']
            })

    result_df = pd.DataFrame(result_rows)
    result_df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"✅ 结果已保存至 {OUTPUT_EXCEL}，有效实验组数：{actual_groups}")
    print("分组概览：")
    for i, group in enumerate(groups):
        print(f"  {assign_ids[i]}: {len(group)} 张图片")

if __name__ == '__main__':
    main()
