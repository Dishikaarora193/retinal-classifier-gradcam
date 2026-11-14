import os
import shutil
import pandas as pd

# 1. Define your paths (update these if your folders are somewhere else)
base_dir = r"C:\Users\Aseem\Downloads\aptos2019-blindness-detection"
csv_path = os.path.join(base_dir, "train.csv")
images_dir = os.path.join(base_dir, "train_images")

# 2. Create the new folders that your AI expects
output_dir = "dataset"
healthy_dir = os.path.join(output_dir, "Healthy")
diseased_dir = os.path.join(output_dir, "Diseased")

os.makedirs(healthy_dir, exist_ok=True)
os.makedirs(diseased_dir, exist_ok=True)

# 3. Read the labels
print("Reading labels from train.csv...")
df = pd.read_csv(csv_path)

# 4. Sort and copy the images
print("Sorting images... this might take a minute.")
for index, row in df.iterrows():
    # The CSV only has the ID, so we add the .png extension
    image_filename = str(row['id_code']) + ".png" 
    source_path = os.path.join(images_dir, image_filename)
    
    if not os.path.exists(source_path):
        continue

    # Label 0 is Healthy. Labels 1, 2, 3, and 4 are different stages of Disease.
    if row['diagnosis'] == 0:
        destination_path = os.path.join(healthy_dir, image_filename)
    else:
        destination_path = os.path.join(diseased_dir, image_filename)
        
    shutil.copy2(source_path, destination_path)

print("Finished! Your images are now sorted into the 'dataset' folder.")