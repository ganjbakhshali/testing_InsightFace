import os
import shutil

def organize_images(input_text_file, input_image_folder, output_folder):
    # Read the text file to get image names and corresponding labels
    with open(input_text_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Split each line to get the image name and label
        image_name, label = line.strip().split()
        label=image_name
        la=label.split("_")
        label=la[0]+"_"+la[1]
        

        # Create a folder for each person based on the label
        person_folder = os.path.join(output_folder, f"{label}")
        os.makedirs(person_folder, exist_ok=True)

        # Build the paths for the source and destination images
        source_image_path = os.path.join(input_image_folder, image_name)
        destination_image_path = os.path.join(person_folder, image_name)

        # copy the image to the corresponding person folder
        shutil.copyfile(source_image_path, destination_image_path)

if __name__ == "__main__":
    # Replace these paths with the actual paths to your text file, image folder, and output folder
    input_text_file = r"C:\Users\Ali\Desktop\testing insightFace\datasets\cplfw\pairs_CPLFW.txt"
    input_image_folder = r"C:\Users\Ali\Desktop\testing insightFace\datasets\cplfw\aligned images"
    output_folder = r"C:\Users\Ali\Desktop\testing insightFace\datasets\cplfw\CPLFW"


    organize_images(input_text_file, input_image_folder, output_folder)
