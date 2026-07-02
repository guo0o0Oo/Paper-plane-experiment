import os, cv2, pandas as pd, numpy as np
from natsort import natsorted

# ==================== 配置 ====================
INPUT_EXCEL   = 'result_final.xlsx'
DATA_FOLDER   = 'data'                     # 原始图片文件夹
OUTPUT_FOLDER = 'composites_with_axes'     # 输出文件夹（覆盖原有图片）
CANVAS_WIDTH  = 2400                       # 统一画布宽度（像素）
CANVAS_HEIGHT = 1600                       # 统一画布高度（像素）
CROP_SIZE     = 120                        # 裁剪正方形的边长（像素）
DRAW_LINES    = True                       # 是否绘制连接线
LINE_COLOR    = (0, 255, 255)              # 连接线颜色（BGR，黄色）
LINE_THICKNESS = 2
AXIS_COLOR    = (255, 255, 255)            # 坐标轴颜色（白色）
TICK_SPACING  = 200                        # 刻度间距（像素）
# ==============================================

def cv_imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

def cv_imwrite(path, img):
    cv2.imencode('.jpg', img)[1].tofile(path)

def crop_center(img, cx, cy, size):
    """以(cx,cy)为中心裁剪size×size的正方形，超出边界用黑色填充"""
    h, w = img.shape[:2]
    half = size // 2
    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    crop = np.zeros((size, size, 3), dtype=img.dtype)
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

    return crop

def draw_axes(canvas, width, height, tick_spacing):
    """在画布左上角绘制坐标系（原点(0,0)，X向右，Y向下）"""
    # 主轴
    cv2.line(canvas, (0, 0), (width-1, 0), AXIS_COLOR, 2)          # X轴（上边界）
    cv2.line(canvas, (0, 0), (0, height-1), AXIS_COLOR, 2)         # Y轴（左边界）
    # 箭头
    cv2.arrowedLine(canvas, (width-20, 0), (width-1, 0), AXIS_COLOR, 2, tipLength=0.3)
    cv2.arrowedLine(canvas, (0, height-20), (0, height-1), AXIS_COLOR, 2, tipLength=0.3)

    # 刻度与标签
    for x in range(0, width, tick_spacing):
        cv2.line(canvas, (x, 0), (x, 5), AXIS_COLOR, 2)
        cv2.putText(canvas, str(x), (x+3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, AXIS_COLOR, 1)
    for y in range(0, height, tick_spacing):
        cv2.line(canvas, (0, y), (5, y), AXIS_COLOR, 2)
        cv2.putText(canvas, str(y), (8, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, AXIS_COLOR, 1)

    # 轴名称
    cv2.putText(canvas, "X (pixels)", (width-80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, AXIS_COLOR, 2)
    cv2.putText(canvas, "Y (pixels)", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, AXIS_COLOR, 2)

def main():
    df = pd.read_excel(INPUT_EXCEL)
    for col in ['实验编号', 'X坐标', 'Y坐标', '图片名称']:
        if col not in df.columns:
            raise ValueError(f"Excel 缺少列: {col}")

    df = df[(df['X坐标'] != '未检测') & (df['Y坐标'] != '未检测')].copy()
    df['X坐标'] = df['X坐标'].astype(float)
    df['Y坐标'] = df['Y坐标'].astype(float)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    half = CROP_SIZE // 2

    grouped = df.groupby('实验编号')
    for exp_id, group in grouped:
        group_sorted = group.copy()
        group_sorted['图片名称'] = pd.Categorical(
            group_sorted['图片名称'],
            categories=natsorted(group_sorted['图片名称'].unique()),
            ordered=True
        )
        group_sorted = group_sorted.sort_values('图片名称').reset_index(drop=True)

        # 创建固定尺寸的黑色画布
        canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

        # 绘制坐标系
        draw_axes(canvas, CANVAS_WIDTH, CANVAS_HEIGHT, TICK_SPACING)

        prev_pt = None
        for idx, row in group_sorted.iterrows():
            img_path = os.path.join(DATA_FOLDER, row['图片名称'])
            if not os.path.exists(img_path):
                continue
            img = cv_imread(img_path)
            if img is None:
                continue

            cx = int(round(row['X坐标']))
            cy = int(round(row['Y坐标']))
            crop = crop_center(img, cx, cy, CROP_SIZE)

            # 粘贴位置（以坐标为中心）
            paste_x = cx - half
            paste_y = cy - half

            # 边界检查（超出部分会被黑色覆盖，这正是我们想要的）
            x_start = max(0, paste_x)
            y_start = max(0, paste_y)
            x_end = min(CANVAS_WIDTH, paste_x + CROP_SIZE)
            y_end = min(CANVAS_HEIGHT, paste_y + CROP_SIZE)

            # 裁剪图区域
            crop_x_start = x_start - paste_x
            crop_y_start = y_start - paste_y
            crop_x_end = crop_x_start + (x_end - x_start)
            crop_y_end = crop_y_start + (y_end - y_start)

            if x_end > x_start and y_end > y_start:
                canvas[y_start:y_end, x_start:x_end] = crop[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

            # 中心点
            center_pt = (cx, cy)
            cv2.circle(canvas, center_pt, 3, (0, 0, 255), -1)

            # 连接线
            if DRAW_LINES and prev_pt is not None:
                cv2.line(canvas, prev_pt, center_pt, LINE_COLOR, LINE_THICKNESS)
            prev_pt = center_pt

            # 序号
            cv2.putText(canvas, str(idx+1), (paste_x+5, paste_y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # 保存，直接覆盖同名文件
        out_path = os.path.join(OUTPUT_FOLDER, f"{exp_id}_axes.jpg")
        cv_imwrite(out_path, canvas)
        print(f"✅ 已保存: {out_path}")

    print("\n全部完成！统一尺寸 2400×1600，已覆盖 composites_with_axes 中的图片。")

if __name__ == '__main__':
    main()
