import os
import zipfile

def compress_images_to_multiple_zips(image_paths, output_zip_prefix, max_size=300 * 1024 * 1024):
    """
    Compress multiple images into multiple zip files without exceeding the specified size for each zip.
    
    :param image_paths: List of paths to the images to be compressed.
    :param output_zip_prefix: Prefix for the output zip files (e.g., 'output' will create output_1.zip, output_2.zip, etc.).
    :param max_size: Maximum size of each zip file in bytes (default is 300 MB).
    """
    zip_index = 1
    total_size = 0
    
    with zipfile.ZipFile(f"{output_zip_prefix}_{zip_index}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, image_path in enumerate(image_paths, start=1):
            if not os.path.isfile(image_path):
                print(f"Warning: {image_path} is not a valid file and will be skipped.")
                continue
            
            # Get the size of the current image
            file_size = os.path.getsize(image_path)
            
            # Check if adding this file would exceed the maximum size
            if total_size + file_size > max_size:
                # Close the current zip file and start a new one
                zipf.close()
                print(f"Completed {output_zip_prefix}_{zip_index}.zip with size: {total_size / (1024 * 1024):.2f} MB")
                zip_index += 1
                total_size = 0
                zipf = zipfile.ZipFile(f"{output_zip_prefix}_{zip_index}.zip", 'w', zipfile.ZIP_DEFLATED)
            
            # Add the file to the zip archive
            zipf.write(image_path, os.path.basename(image_path))
            total_size += file_size
            print(f"[{i}/{len(image_paths)}] Added {image_path} to {output_zip_prefix}_{zip_index}.zip, current zip size: {total_size / (1024 * 1024):.2f} MB")
        
        # Print completion message for the last zip file
        print(f"Completed {output_zip_prefix}_{zip_index}.zip with size: {total_size / (1024 * 1024):.2f} MB")

# Example usage
if __name__ == "__main__":
    # Assuming you have a directory full of images
    image_directory = 'J:/Data/Test/543/yellow_river/crop/river_ice/2025'
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', 'tif'))]
    
    output_zip_prefix = 'J:/Data/Test/543/yellow_river/crop/river_ice/package/2025'
    
    compress_images_to_multiple_zips(image_files, output_zip_prefix)



