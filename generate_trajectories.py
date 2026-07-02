import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from natsort import natsorted
import numpy as np
from scipy.interpolate import make_interp_spline

# ---------- 中文字体设置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']      # 或 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42                 # 嵌入字体
# ---------------------------------

# ==================== 配置 ====================
INPUT_EXCEL  = 'result_final.xlsx'      # 清洗后的结果表
OUTPUT_PDF   = 'trajectories.pdf'       # 输出的多页 PDF
# ==============================================

def plot_experiment(exp_id, df_group):
    """为单个实验绘制轨迹图，返回 matplotlib figure"""
    # 按图片名自然排序（保证时间顺序）
    df_sorted = df_group.copy()
    df_sorted['图片名称'] = pd.Categorical(
        df_sorted['图片名称'],
        categories=natsorted(df_sorted['图片名称'].unique()),
        ordered=True
    )
    df_sorted = df_sorted.sort_values('图片名称').reset_index(drop=True)

    # 提取坐标
    x = df_sorted['X坐标'].astype(float).values
    y = df_sorted['Y坐标'].astype(float).values
    n = len(x)

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制散点（原始数据点）
    ax.scatter(x, y, c='red', s=30, zorder=5, label='数据点')

    # 如果数据点足够，绘制平滑曲线
    if n >= 4:
        # 用参数 t（0~1）分别对 x 和 y 做三次样条插值
        t = np.linspace(0, 1, n)
        # 插值到更密集的 t
        t_new = np.linspace(0, 1, 300)
        try:
            spl_x = make_interp_spline(t, x, k=min(3, n-1))
            spl_y = make_interp_spline(t, y, k=min(3, n-1))
            x_new = spl_x(t_new)
            y_new = spl_y(t_new)
            ax.plot(x_new, y_new, 'b-', linewidth=1.5, label='平滑曲线')
        except Exception as e:
            # 如果插值失败（如点重复等），退化为直线连接
            ax.plot(x, y, 'b-', linewidth=1.5, label='折线连接')
    elif n >= 2:
        # 点太少，直接连线
        ax.plot(x, y, 'b-', linewidth=1.5, label='折线连接')

    # 设置标题和坐标轴
    ax.set_title(f'实验 {exp_id} 纸飞机轨迹', fontsize=14)
    ax.set_xlabel('X 坐标 (像素)')
    ax.set_ylabel('Y 坐标 (像素)')
    # Y 轴翻转（通常图像 Y 从上到下增加，轨迹图中常希望上为正，这里可选择翻转）
    # 因为纸飞机向下飞，Y 坐标增大，如果想保留原始图像坐标方向则不翻转
    ax.invert_yaxis()   # 注释掉若不希望翻转
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axis('equal')    # 等比例坐标，避免拉伸变形

    return fig

def main():
    # 读取数据
    df = pd.read_excel(INPUT_EXCEL)
    print(f"读取数据：{len(df)} 行")

    # 分组
    grouped = df.groupby('实验编号')
    exp_ids = natsorted(grouped.groups.keys())
    print(f"共 {len(exp_ids)} 个实验")

    # 创建 PDF 文件
    with matplotlib.backends.backend_pdf.PdfPages(OUTPUT_PDF) as pdf:
        for idx, exp_id in enumerate(exp_ids):
            group = grouped.get_group(exp_id)
            fig = plot_experiment(exp_id, group)
            pdf.savefig(fig)
            plt.close(fig)
            if (idx + 1) % 10 == 0:
                print(f"已生成 {idx+1}/{len(exp_ids)} 个实验图表...")

    print(f"✅ 所有轨迹图已保存至：{OUTPUT_PDF}")

if __name__ == '__main__':
    main()
