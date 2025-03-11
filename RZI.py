import os

def delete_zone_identifier_files(directory):
    """Recursively deletes all .zone.identifier files in the given directory.
    This functions only purpouse is to delet all files tha WSL creates when downloading a file in windows and moving them
    to a WSL folder
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith("Zone.Identifier"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    if os.path.isdir(directory):
        delete_zone_identifier_files(directory)
    else:
        print("Invalid directory path.")
