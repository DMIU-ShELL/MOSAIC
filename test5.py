import os
import shutil

def move_video_files(source_dir, destination_dir):
    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Walk through all subdirectories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".video.mp4"):
                full_path = os.path.join(root, file)
                dest_path = os.path.join(destination_dir, file)

                # Avoid overwriting files with the same name
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(dest_path):
                        new_filename = f"{base}_{counter}{ext}"
                        dest_path = os.path.join(destination_dir, new_filename)
                        counter += 1

                print(f"Moving: {full_path} -> {dest_path}")
                shutil.move(full_path, dest_path)

# Example usage
source_directory = "./log/FINAL/mctgraph/"
destination_directory = "./videos2/"

move_video_files(source_directory, destination_directory)