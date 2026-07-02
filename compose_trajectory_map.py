import os, cv2, pandas as pd, numpy as np
from natsort import natsorted

# ==================== 配置 ====================
INPUT_EXCEL   = 'result_final.xlsx'
DATA_FOLDER   = 'data'                    # 原始图片文件夹
OUTPUT_FOLDER = 'composites'              # 输出合成图的文件夹
CROP_SIZE     = 120                       # 裁剪的正方形边长（像素），会以此为中心
DRAW_LINES    = True                      # 是否在裁剪图之间画连接线（显示轨迹）
LINE_COLOR    = (0, 255, 255)             # 连接线颜色（BGR），默认黄色
LINE_THICKNESS = 2
# ==============================================

def cv_imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

def cv_imwrite(path, img):
    cv2.imencode('.jpg', img)[1].tofile(path)

def crop_center(img, cx, cy, size):
    """以 (cx, cy) 为中心裁剪 size×size 的正方形，超出边界的部分用黑色填充"""
    h, w = img.shape[:2]
    half = size // 2
    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    # 计算原图中的有效区域
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    # 创建黑色画布
    crop = np.zeros((size, size, 3), dtype=img.dtype)

    # 计算粘贴位置
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

    return crop

def main():
    df = pd.read_excel(INPUT_EXCEL)
    # 检查必要列
    for col in ['实验编号', 'X坐标', 'Y坐标', '图片名称']:
        if col not in df.columns:
            raise ValueError(f"Excel 缺少列: {col}")

    # 过滤无效坐标
    df = df[(df['X坐标'] != '未检测') & (df['Y坐标'] != '未检测')].copy()
    df['X坐标'] = df['X坐标'].astype(float)
    df['Y坐标'] = df['Y坐标'].astype(float)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 按实验编号分组
    grouped = df.groupby('实验编号')
    for exp_id, group in grouped:
        # 组内按文件名自然排序，保证时间顺序
        group_sorted = group.copy()
        group_sorted['图片名称'] = pd.Categorical(
            group_sorted['图片名称'],
            categories=natsorted(group_sorted['图片名称'].unique()),
            ordered=True
        )
        group_sorted = group_sorted.sort_values('图片名称').reset_index(drop=True)

        # 提取所有坐标点
        xs = group_sorted['X坐标'].values
        ys = group_sorted['Y坐标'].values

        if len(xs) == 0:
            continue

        # 计算画布大小：覆盖所有裁剪框的外接矩形
        half = CROP_SIZE // 2
        min_x = int(np.min(xs) - half)
        max_x = int(np.max(xs) + half)
        min_y = int(np.min(ys) - half)
        max_y = int(np.max(ys) + half)
        canvas_w = max_x - min_x
        canvas_h = max_y - min_y

        # 创建黑色画布
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # 用于画连接线的上一个中心点在画布上的坐标
        prev_pt = None

        for idx, row in group_sorted.iterrows():
            # 读取原图
            img_path = os.path.join(DATA_FOLDER, row['图片名称'])
            if not os.path.exists(img_path):
                print(f"⚠️ 找不到图片: {img_path}")
                continue
            img = cv_imread(img_path)
            if img is None:
                continue

            cx = int(round(row['X坐标']))
            cy = int(round(row['Y坐标']))

            # 裁剪中心区域
            crop = crop_center(img, cx, cy, CROP_SIZE)

            # 计算裁剪图在画布上的左上角坐标（使裁剪图的中心对齐坐标点）
            paste_x = cx - min_x - half
            paste_y = cy - min_y - half

            # 确保粘贴区域不越界
            # 计算实际可粘贴区域（裁剪图尺寸不变，但要完全在画布内）
            if (paste_x >= 0 and paste_y >= 0 and
                paste_x + CROP_SIZE <= canvas_w and paste_y + CROP_SIZE <= canvas_h):
                canvas[paste_y:paste_y+CROP_SIZE, paste_x:paste_x+CROP_SIZE] = crop
            else:
                # 如果越界，可只粘贴重叠部分（为简化，这里略过，一般不会越界）
                pass

            # 在画布上标记中心点（可选）
            center_on_canvas = (cx - min_x, cy - min_y)
            cv2.circle(canvas, center_on_canvas, 3, (0, 0, 255), -1)

            # 画连接线
            if DRAW_LINES and prev_pt is not None:
                cv2.line(canvas, prev_pt, center_on_canvas, LINE_COLOR, LINE_THICKNESS)
            prev_pt = center_on_canvas

            # 可以在裁剪图上叠加顺序数字（可选）
            cv2.putText(canvas, str(idx+1), (paste_x+5, paste_y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # 保存合成图
        out_path = os.path.join(OUTPUT_FOLDER, f"{exp_id}_composite.jpg")
        cv_imwrite(out_path, canvas)
        print(f"✅ 已保存: {out_path}  (绘制了 {len(group_sorted)} 个裁剪块)")

    print("\n全部完成！")

if __name__ == '__main__':
    main()
