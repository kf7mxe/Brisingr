
# check if any of the files in the folder are duplicates if they are not duplicates copy them to a different folder

# Importing Libraries
import os
from pathlib import Path
from filecmp import cmp
import shutil

folder_to_check = ''
folder_to_copy = ''



# Function to check if the files are duplicates
files_in_folder = os.listdir(folder_to_check)
for file in files_in_folder:
    is_duplicate = False
    for file2 in files_in_folder:
        if cmp(Path(folder_to_check + file), Path(folder_to_check + file2)):
            is_duplicate = True
            break
    if not is_duplicate:
        shutil.copy(folder_to_check + file, folder_to_copy)

