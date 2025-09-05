import csv
import numpy as np
from pathlib import Path
from logger import info, debug

work_space = Path(__file__).parent.parent

def readCSV(directory_path, file_name):
    csv_path = work_space / directory_path / file_name

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"File {file_name} is not a CSV file")

    info(f"Start to read points from {csv_path}.")

    points = []
    reflectivities = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            try:
                point = [float(row[8]), float(row[9]), float(row[10])]
                reflectivity = float(row[11])
                points.append(point)
                reflectivities.append(reflectivity)
            except ValueError as e:
                debug(f"Skipping invalid row {row}: {e}")

    points = np.asarray(points, dtype=np.float32)
    reflectivities = np.asarray(reflectivities, dtype=np.float32)
    info(f"All {len(points)} points has been read from {csv_path}.")
    return points, reflectivities

def writeCSV(directory_path, file_name, points: list):
    if len(points) == 0:
        raise RuntimeError("You have not choose any points")
        
    csv_path = work_space / directory_path / file_name

    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open(mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'z'])

        for point in points:
            if len(point) != 3:
                raise RuntimeError(f"Point size should be 3, but {len(point)} has been given")

            writer.writerow(point)

    info(f"All {len(points)} points have been written to {csv_path}.")