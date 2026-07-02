import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from natsort import natsorted
import numpy as np
from scipy.interpolate import make_interp_spline
import re

# ==================== 配置 ====================
INPUT_EXCEL  = 'result_final.xlsx'
OUTPUT_C_PER_AB = 'compare_c_per_ab.pdf'       # 同a同b下不同c对比
OUTPUT_B_PER_A  = 'compare_b_per_a.pdf'        # 同a下不同b对比（同b同色，实线）
OUTPUT_A_PER_B  = 'compare_a_per_b.pdf'        # 同b下不同a对比（同a同色，实线）
# ==============================================

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

group_colors = ['#FF0000','#00BFFF','#32CD32','#FFA500','#8A2BE2']

def parse_id(exp_id):
    match = re.match(r'(\d)0(\d)0(\d)', str(exp_id))
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None

def smooth_trajectory(x, y, min_points=4):
    n = len(x)
    if n < min_points:
        return x, y
    t = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, 300)
    try:
        spl_x = make_interp_spline(t, x, k=min(3, n-1))
        spl_y = make_interp_spline(t, y, k=min(3, n-1))
        return spl_x(t_new), spl_y(t_new)
    except:
        return x, y

def plot_single_experiment(ax, df_exp, label, color):
    df_sorted = df_exp.copy()
    df_sorted['图片名称'] = pd.Categorical(
        df_sorted['图片名称'],
        categories=natsorted(df_sorted['图片名称'].unique()),
        ordered=True
    )
    df_sorted = df_sorted.sort_values('图片名称').reset_index(drop=True)
    x = df_sorted['X坐标'].astype(float).values
    y = df_sorted['Y坐标'].astype(float).values

    if len(x) == 0:
        return
    if len(x) == 1:
        ax.scatter(x[0], y[0], color=color, marker='o', s=30, zorder=5, label=label)
        return

    if len(x) >= 4:
        xs, ys = smooth_trajectory(x, y)
    else:
        xs, ys = x, y

    ax.plot(xs, ys, color=color, linestyle='-', linewidth=1.5, label=label)
    ax.scatter(x[0], y[0], color=color, marker='o', s=20, zorder=5)
    ax.scatter(x[-1], y[-1], color=color, marker='s', s=20, zorder=5)

def main():
    df = pd.read_excel(INPUT_EXCEL)
    df['a'] = df['实验编号'].apply(lambda x: parse_id(x)[0])
    df['b'] = df['实验编号'].apply(lambda x: parse_id(x)[1])
    df['c'] = df['实验编号'].apply(lambda x: parse_id(x)[2])
    df = df.dropna(subset=['a','b','c']).copy()
    df['实验编号'] = df['实验编号'].astype(str)
    df['a'] = df['a'].astype(int)
    df['b'] = df['b'].astype(int)
    df['c'] = df['c'].astype(int)

    all_exp_ids = natsorted(df['实验编号'].unique())
    exp_to_df = {eid: df[df['实验编号'] == eid] for eid in all_exp_ids}

    # 1. 同a同b不同c
    print("生成 compare_c_per_ab.pdf ...")
    with matplotlib.backends.backend_pdf.PdfPages(OUTPUT_C_PER_AB) as pdf1:
        for a in sorted(df['a'].unique()):
            for b in sorted(df['b'].unique()):
                group_ab = df[(df['a']==a) & (df['b']==b)]
                if group_ab.empty: continue
                c_vals = sorted(group_ab['c'].unique())
                these_ids = [f"{a}0{b}0{c}" for c in c_vals]
                colors = [group_colors[i % len(group_colors)] for i in range(len(c_vals))]

                fig, ax = plt.subplots(figsize=(8,6))
                ax.set_title(f'实验组 {a}0{b} 各次飞行对比')
                ax.set_xlabel('X (像素)'); ax.set_ylabel('Y (像素)')
                ax.invert_yaxis(); ax.grid(True, linestyle='--', alpha=0.6)

                for eid, col in zip(these_ids, colors):
                    if eid in exp_to_df:
                        plot_single_experiment(ax, exp_to_df[eid], eid, col)

                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=7)
                else:
                    ax.text(0.5, 0.5, '无有效轨迹', transform=ax.transAxes,
                            ha='center', va='center', fontsize=14, color='gray')
                pdf1.savefig(fig); plt.close(fig)
    print("✅ compare_c_per_ab.pdf 完成")

    # 2. 同a不同b对比
    print("生成 compare_b_per_a.pdf ...")
    with matplotlib.backends.backend_pdf.PdfPages(OUTPUT_B_PER_A) as pdf2:
        for a in sorted(df['a'].unique()):
            ids_a = [eid for eid in all_exp_ids if parse_id(eid)[0] == a]
            if not ids_a: continue
            b_vals = sorted({parse_id(eid)[1] for eid in ids_a})
            b_color = {b: group_colors[i % len(group_colors)] for i, b in enumerate(b_vals)}

            fig, ax = plt.subplots(figsize=(10,7))
            ax.set_title(f'实验组 a={a} 所有飞行轨迹 (同b同色)')
            ax.set_xlabel('X (像素)'); ax.set_ylabel('Y (像素)')
            ax.invert_yaxis(); ax.grid(True, linestyle='--', alpha=0.6)

            for eid in ids_a:
                _, b, c = parse_id(eid)
                plot_single_experiment(ax, exp_to_df[eid], eid, b_color[b])
            ax.legend(fontsize=6)
            pdf2.savefig(fig); plt.close(fig)
    print("✅ compare_b_per_a.pdf 完成")

    # 3. 同b不同a对比
    print("生成 compare_a_per_b.pdf ...")
    with matplotlib.backends.backend_pdf.PdfPages(OUTPUT_A_PER_B) as pdf3:
        for b in sorted(df['b'].unique()):
            ids_b = [eid for eid in all_exp_ids if parse_id(eid)[1] == b]
            if not ids_b: continue
            a_vals = sorted({parse_id(eid)[0] for eid in ids_b})
            a_color = {a: group_colors[i % len(group_colors)] for i, a in enumerate(a_vals)}

            fig, ax = plt.subplots(figsize=(10,7))
            ax.set_title(f'实验组 b={b} 所有飞行轨迹 (同a同色)')
            ax.set_xlabel('X (像素)'); ax.set_ylabel('Y (像素)')
            ax.invert_yaxis(); ax.grid(True, linestyle='--', alpha=0.6)

            for eid in ids_b:
                a, _, c = parse_id(eid)
                plot_single_experiment(ax, exp_to_df[eid], eid, a_color[a])
            ax.legend(fontsize=6)
            pdf3.savefig(fig); plt.close(fig)
    print("✅ compare_a_per_b.pdf 完成")

if __name__ == '__main__':
    main()
