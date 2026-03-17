import cv2
import numpy as np

# ===================== 【参数配置】修改这里 =====================
# 棋盘格内角点尺寸（根据你的棋盘格修改！！！）
PATTERN_SIZE = (9, 6)  # 横向9个角点，纵向6个角点
# 你的图片路径（替换成自己的照片路径）
IMAGE_PATH = "res/chess.jpg"
# =================================================================

# 1. 读取图片
img = cv2.imread(IMAGE_PATH)
# 转换为灰度图（角点检测需要灰度图）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("开始寻找角点...")

# 2. 检测棋盘格角点
# ret：是否检测到角点；corners：检测到的角点坐标
ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

# 3. 如果检测成功，优化角点精度并绘制
if ret:
    # 亚像素级优化（让角点坐标更精确）
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # 在原图上绘制角点（彩色连线+圆点）
    cv2.drawChessboardCorners(img, PATTERN_SIZE, corners_subpix, ret)
    
    print(f"✅ 成功检测到 {PATTERN_SIZE[0]}x{PATTERN_SIZE[1]} 棋盘格角点！")
else:
    print("❌ 未检测到棋盘格角点，请检查：")
    print("   1. 棋盘格尺寸 PATTERN_SIZE 是否正确")
    print("   2. 图片是否清晰、无遮挡、棋盘格完整")

# 4. 显示结果
cv2.imshow("棋盘格角点检测结果", img)

# 等待按键后关闭窗口
cv2.waitKey(0)
# 保存带角点的结果图
cv2.imwrite("chessboard_with_corners.jpg", img)

# 释放资源
cv2.destroyAllWindows()