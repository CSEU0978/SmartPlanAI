import os
from pathlib import Path
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import io


file_path = Path("C:" + os.sep + "Users" + os.sep + "suran" + os.sep + "Desktop" + os.sep + "School" + os.sep +
                "1_UNIVERSITY" + os.sep + "BENNETT" + os.sep + "6thSem" + os.sep + "IVP" + os.sep + "SmartPlanAI" + os.sep +
                "Data")

parq1_dir = Path(str(file_path) + os.sep + "train-00000-of-00008-0a83f93dc11798d2.parquet")
parq2_dir = Path(str(file_path) + os.sep + "train-00001-of-00008-2bc1ace2906b29d6.parquet")
table = pq.read_table(parq1_dir)
schema = table.schema
print(schema)

df = table.to_pandas()
print(df.columns)

#df = pd.read_parquet(file_path)
# Create the main database directory
database_dir = Path(str(file_path) + os.sep + "Database")
os.makedirs(database_dir, exist_ok=True)

# List of subfolders for different types of data
subfolders = ['plans', 'walls', 'colors', 'footprints', 'plan_captions']

# Create subdirectories for each type of data
for subfolder in subfolders:
    subfolder_path = os.path.join(database_dir, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)

# Iterate through each row of the DataFrame
for index, row in df.iterrows():
    # Get the index value from the 'indices' column
    idx = row['indices']

    # Iterate through each subfolder type
    for subfolder in subfolders[:-1]:  # Exclude 'description' subfolder
        # Check if the column contains image data as a dictionary
        if isinstance(row[subfolder], dict):
            # Extract the image data from the dictionary
            image_data = row[subfolder]['bytes']

            # Save the image to the corresponding subfolder using the index value
            image = Image.open(io.BytesIO(image_data))
            image_path = os.path.join(database_dir, subfolder, f"{idx}.png")  # Adjust file extension as needed
            image.save(image_path)

    # Save the description to the 'description' subfolder using the index value
    description_path = os.path.join(database_dir, 'plan_captions', f"{idx}.txt")
    with open(description_path, 'w') as file:
        file.write(row['plan_captions'])

# Show the first few rows of the DataFrame to verify
print(df.head())