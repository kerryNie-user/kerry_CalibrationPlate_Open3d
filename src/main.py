# -*- coding: utf-8 -*-
# main.py  — Open3D 0.19, 鼠标射线检测 OBB
import argparse
import logger
import CsvIO
from CalibrationApplicator import CalibrationApplicator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibration Plate App')
    parser.add_argument('--input',  default='res/map.csv',
                        help='输入点云文件路径，支持 .csv 和 .ply（默认：res/map.csv）')
    parser.add_argument('--output', default='output/corner.csv',
                        help='输出角点 CSV 文件路径（默认：output/corner.csv）')
    args = parser.parse_args()

    # 设置调试输出等级
    logger.set_log_level(level=logger.SUGGESTION)
    logger.info("<<< Calibration Plate App >>>")

    # 根据文件扩展名自动选择读取方式（.csv 或 .ply）
    points, reflectivities = CsvIO.readPoints(args.input)

    # 运行程序
    application = CalibrationApplicator(points, reflectivities)
    application.app_.run()

    # 从程序中拿到标定点
    points = application.get_corner_points()

    # 向 csv 文件写入标定点
    CsvIO.writeCSV(args.output, points)
    logger.info(">>> Calibration Plate App <<<")