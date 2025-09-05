# main.py  — Open3D 0.19, 鼠标射线检测 OBB
import logger
import CsvIO
from ObbSelectApplicator import ObbSelectApplicator

if __name__ == "__main__":
    logger.setup_logger(level=logger.SUGGESTION)
    logger.info("<<< Calibration Plate App >>>")

    points, reflectivities = CsvIO.readCSV("res", "map.csv")

    application = ObbSelectApplicator(points, reflectivities)
    application.run()
    points = application.get_corner_points()

    CsvIO.writeCSV("output", "corner.csv", points)
    logger.info(">>> Calibration Plate App <<<")