"""
This script generates a CSV file containing the labels for the dataset. 
It iterates through the dataset folder structure, which is organized as follows:
Dataset/
    neutral/
        with/
            image1.bmp
            ...
        without/
            image2.bmp
            ...
    paper/
        with/
            image4.bmp
            ...
        without/
            image5.bmp
            ...
    rock/
        with/
            image7.bmp
            ...
        without/
            image8.bmp
            ...
    scissors/
        with/
            image10.bmp
            ...
        without/
            image11.bmp
            ...

The generated CSV file will have the following format:
filename,gesture,accessory
"""

from pathlib import Path
import csv

rows = []

for gesture_dir in Path("dataset").iterdir():
    if not gesture_dir.is_dir():
        continue

    for accessory_dir in gesture_dir.iterdir():
        if not accessory_dir.is_dir():
            continue

        accessory = (
            "present"
            if accessory_dir.name == "with"
            else "absent"
        )

        for image in accessory_dir.glob("*.bmp"):
            rows.append([
                str(image),
                gesture_dir.name,
                accessory
            ])

with open("dataset/labels.csv","w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename","gesture","accessory"])
    writer.writerows(rows)