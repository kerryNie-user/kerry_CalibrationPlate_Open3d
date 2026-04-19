import os
import cv2

# ===================== 【参数配置】修改这里 =====================
# 棋盘格内角点尺寸（根据你的棋盘格修改！！！）
PATTERN_SIZE = (9, 6)  # 横向9个角点，纵向6个角点
# 你的图片路径（替换成自己的照片路径）
IMAGE_PATH = "res/chess.jpg"
POINTS_PATH = "output/camera_corner.csv"
# =================================================================

def try_detect(gray_image, pattern_size):
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FILTER_QUADS
    )

    candidates = [
        gray_image,
        cv2.equalizeHist(gray_image),
        cv2.GaussianBlur(gray_image, (5, 5), 0),
    ]

    for candidate in candidates:
        ret, corners = cv2.findChessboardCorners(candidate, pattern_size, flags)
        if ret:
            return True, corners, candidate, "经典算法"

    if hasattr(cv2, "findChessboardCornersSB"):
        for candidate in candidates:
            ret, corners = cv2.findChessboardCornersSB(candidate, pattern_size, flags)
            if ret:
                return True, corners, candidate, "SB算法"

    return False, None, None, None


img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"图片读取失败，请检查路径: {IMAGE_PATH}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("开始寻找角点...")

ret, corners, used_gray, method_name = try_detect(gray, PATTERN_SIZE)

detected_pattern = PATTERN_SIZE
if not ret:
    swapped_pattern = (PATTERN_SIZE[1], PATTERN_SIZE[0])
    ret, corners, used_gray, method_name = try_detect(gray, swapped_pattern)
    if ret:
        detected_pattern = swapped_pattern

if ret:
    if method_name == "经典算法":
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.0001)
        corners = cv2.cornerSubPix(used_gray, corners, (11, 11), (-1, -1), criteria)

    cv2.drawChessboardCorners(img, detected_pattern, corners, ret)
    print(f"✅ 成功检测到 {detected_pattern[0]}x{detected_pattern[1]} 棋盘格角点（{method_name}）！")
    output_dir = os.path.dirname(POINTS_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(POINTS_PATH, "w", encoding="utf-8") as f:
        f.write("x,y\n")
        for p in corners.reshape(-1, 2):
            x, y = float(p[0]), float(p[1])
            line = f"{x:.6f},{y:.6f}"
            print(line)
            f.write(line + "\n")
    print(f"角点坐标已保存到: {POINTS_PATH}")
    if detected_pattern != PATTERN_SIZE:
        print(f"⚠️ 你当前配置是 {PATTERN_SIZE}，但实际检测到的是 {detected_pattern}，建议修改 PATTERN_SIZE。")
else:
    print("❌ 未检测到棋盘格角点，请检查：")
    print("   1. PATTERN_SIZE 是否是“内角点数”（不是格子数）")
    print("   2. 图片是否清晰、无遮挡、棋盘格完整")
    print("   3. 棋盘是否有明显反光、过曝或过暗")

cv2.imwrite("output/chessboard_with_corners.jpg", img)
cv2.destroyAllWindows()
