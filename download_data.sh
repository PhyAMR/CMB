#!/bin/bash

# Configuration
ZIP_URL="http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_fullGrid_R3.01.zip"
ZIP_NAME="COM_CosmoParams_fullGrid_R3.01.zip"
TARGET_DIR="Planck_Data"

# 1. Create the Planck_Data folder if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "--- Starting Planck CosmoParams Download ---"

# 2. Download the file
# -O saves it with the correct filename
# -c allows resuming if the connection breaks
echo "Downloading $ZIP_NAME..."
wget -c "$ZIP_URL" -O "$TARGET_DIR/$ZIP_NAME"

# 3. Extract the data
echo "Extracting files into $TARGET_DIR..."
# -n prevents overwriting files that already exist (saves time if re-running)
unzip -n "$TARGET_DIR/$ZIP_NAME" -d "$TARGET_DIR"

echo "--- Done! Your data is in the $TARGET_DIR folder. ---"

echo "Note: If you need to re-run this script, it will skip downloading and extracting files that already exist."

echo "If you want to force re-download or re-extraction, delete the existing files in $TARGET_DIR and run this script again."

echo "To donload other Planck Products like maps to use the maps plots, you should download the maps from https://pla.esac.esa.int/pla/#home and put them in the maps/ folder."