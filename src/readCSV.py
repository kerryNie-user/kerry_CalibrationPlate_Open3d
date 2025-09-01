import csv
import numpy as np
from pathlib import Path
from logger import info

def readCSV(directory_path, file_name):
    work_space = Path(__file__).parent.parent
    csv_path = work_space / directory_path / file_name

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"File {file_name} is not a CSV file")

    info(f"Start to read points from {csv_path}.")

    points, reflectivities = [], []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            points.append([float(row[8]), float(row[9]), float(row[10])])
            reflectivities.append(float(row[11]))

    points = np.asarray(points, dtype=np.float32)
    reflectivities = np.asarray(reflectivities, dtype=np.float32)
    info(f"All {len(points)} points has been read.")
    return points, reflectivities

def writeCSV(directory_path, file_name, points: list):
    if len(points) == 0:
        raise RuntimeError("You have not choose any points")
        
    work_space = Path(__file__).parent.parent
    csv_path = work_space / directory_path / file_name

    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    info(f"Start to write points and labels to {csv_path}.")

    with csv_path.open(mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'z'])

        for point in points:
            if len(point) != 3:
                raise RuntimeError(f"Point size should be 3, but {len(point)} has been given")

            writer.writerow(point)

    info(f"All {len(points)} points with labels have been written to {csv_path}.")