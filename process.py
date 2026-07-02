import cv2, os, pandas as pd, numpy as np, easyocr
from ultralytics import YOLO

# ==================== 配置区域 ====================
INPUT_FOLDER   = 'data'                                    # 原始 800 张图片文件夹
OUTPUT_FOLDER  = 'output'                                  # 保存带红点的图片
EXCEL_NAME     = 'result.xlsx'                             # 最终结果表
MODEL_PATH     = 'datasets/paper_plane/runs/detect/plane_detect/weights/best.pt'  # 你的最佳模型
# ==================================================

# 中文路径兼容
def cv_imread(path): return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
def cv_imwrite(path, img): cv2.imencode('.jpg', img)[1].tofile(path)

# ---------- 全局加载模型 (只加载一次) ----------
print("⏳ 加载 YOLO 模型...")
model = YOLO(MODEL_PATH)
print("✅ YOLO 模型加载完成")

print("⏳ 加载 EasyOCR 模型（仅一次）...")
reader = easyocr.Reader(['en'], gpu=False)
print("✅ OCR 模型加载完成")

# ---------- OCR 识别函数（复用之前的增强版）----------
def detect_number(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    h, w = binary.shape
    binary[:int(h*0.1), :] = 0
    binary[-int(h*0.1):, :] = 0
    binary[:, :int(w*0.05)] = 0
    binary[:, -int(w*0.05)] = 0

    results = reader.readtext(binary, allowlist='0123456789', detail=0)
    candidates = [r for r in results if len(r) >= 3]
    if candidates:
        return max(candidates, key=len)
    return "未知编号"

# ---------- 主流程 ----------
def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ 找不到文件夹：{INPUT_FOLDER}")
        return
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    images.sort()
    print(f"📷 共发现 {len(images)} 张图片，开始处理...\n")

    results_data = []   # 存入 Excel

    for idx, img_name in enumerate(images):
        img_path = os.path.join(INPUT_FOLDER, img_name)
        frame = cv_imread(img_path)
        if frame is None:
            print(f"⚠️ 读取失败：{img_name}")
            continue
        h, w, _ = frame.shape

        # 1. OCR 编号（右下角区域）
        roi = frame[int(h*0.7):h, int(w*0.3):w]
        number = detect_number(roi)

        # 2. YOLO 检测飞机
        results = model(frame, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            # 取置信度最高的框（若有多只，可修改逻辑）
            best = max(boxes, key=lambda b: b.conf)
            x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
            conf = float(best.conf[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 画红点
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            # 在左上角显示编号
            cv2.putText(frame, number, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            results_data.append([number, cx, cy, conf])
        else:
            cv2.putText(frame, number + " [未检测到飞机]", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            results_data.append([number, "未检测", "未检测", 0.0])

        # 保存带标注的图片
        cv_imwrite(os.path.join(OUTPUT_FOLDER, img_name), frame)

        if (idx + 1) % 50 == 0:
            print(f"⏳ 已处理 {idx+1}/{len(images)} 张...")

    # 写入 Excel
    df = pd.DataFrame(results_data, columns=['实验编号', 'X坐标', 'Y坐标', '置信度'])
    df.to_excel(EXCEL_NAME, index=False)
    print(f"\n✅ 全部完成！共处理 {len(results_data)} 张图片。")
    print(f"📁 标注图片保存在：{OUTPUT_FOLDER}")
    print(f"📊 结果表格：{EXCEL_NAME}")

if __name__ == '__main__':
    main()
