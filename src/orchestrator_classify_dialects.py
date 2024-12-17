import os
import glob
import sys

# Define source OSCAR directories
source_pattern = "/gpfs/projects/bsc88/data/01-raw-collections/colossal-oscar-*/filtered/*_LANG*parquet"
alt_source_pattern = "/gpfs/projects/bsc88/data/01-raw-collections/colossal-oscar-*/extracted/*/*LANG*jsonl"

# Languages of interest
#languages = ["es", "ca", "gl", "eu", "pt", "oc", "an", "ast"]
languages = ["ca"]

def empty_path(output_path):
    if os.path.exists(output_path):
        os.remove(output_path)

# Function to move and rename files
def move_files(output_folder):

    # Traverse through each language folder in the input directory
    for lang in languages:
        # Use glob to find all matching files
        files = glob.glob(source_pattern.replace("LANG", lang))
        print(f"{len(files)} found for {lang} using {source_pattern.replace("LANG", lang)}")
        if len(files) == 0:
            print("Trying again.")
            files = glob.glob(alt_source_pattern.replace("LANG", lang))
            print(f"{len(files)} found for {lang} using {alt_source_pattern.replace("LANG", lang)}")
        
        i = 0

        # Move and rename each file
        for file in files:
            colossal_oscar_date = file.split("/")[6].split("_")[0]
            # Extract filename from the source path
            filename = os.path.basename(file)
            extension = filename.split(".")[-1]
            new_filename = f"{colossal_oscar_date}_{lang}_part{i}.jsonl"

            # Define the full path for the output file
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            greasy = open(greasy_path, "a+")
            greasy.write('\n')
            greasy.write('[@ {} @]'.format(os.getcwd()))
            greasy.write(" export INPUT_FILE='{}'".format(file))
            greasy.write(" export OUTPUT_DIR='{}'".format(output_folder))
            greasy.write(' && bash /gpfs/projects/bsc88/data/09-scripts/catalan-dialects-classifier/classify_dialects.sh')

            i += 1
    
    print(f'Shell script written to {greasy_path}. All copy commands generated.')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 orchestrator_classify_dialects.py <greasy_path> <output_folder>")
        sys.exit(1)

    greasy_path = sys.argv[1]
    output_folder = sys.argv[2]

    empty_path(greasy_path)
    move_files(output_folder)
    
