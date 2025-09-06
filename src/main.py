# -*- coding: utf-8 -*-
# main.py  — Open3D 0.19, 鼠标射线检测 OBB
import logger
import CsvIO
from CalibrationApplicator import CalibrationApplicator

if __name__ == "__main__":
    # 设置调试输出等级
    logger.set_log_level(level=logger.SUGGESTION)
    logger.info("<<< Calibration Plate App >>>")

    # 从 csv 文件读取点位和点反射率
    points, reflectivities = CsvIO.readCSV("res", "map.csv")

    # 运行程序
    application = CalibrationApplicator(points, reflectivities)
    application.app_.run()

    # 从程序中拿到标定点
    points = application.get_corner_points()

    # 向 csv 文件写入标定点
    CsvIO.writeCSV("output", "corner.csv", points)
    logger.info(">>> Calibration Plate App <<<")