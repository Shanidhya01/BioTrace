import shutil
import os

processed_dir = "processed_data"

# Delete the processed_data folder if it exists
if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)
    print(f"Deleted '{processed_dir}' folder. Ready to train from scratch.")
else:
    print(f"No '{processed_dir}' folder found. Ready to train from scratch.")
