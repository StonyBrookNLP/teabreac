#!/bin/bash

set -e
set -x

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.
pip install gdown

ZIP_NAME="processed_target_datasets.zip"

# URL: https://drive.google.com/file/d/1gPb9A32mwFyvjKcd3NPKRU1eFj8lBWGC/view?usp=sharing
gdown --id 1gPb9A32mwFyvjKcd3NPKRU1eFj8lBWGC --output $ZIP_NAME
unzip $(basename $ZIP_NAME)
rm $ZIP_NAME

# TODO: prevent these from zipping in.
rm -rf __MACOSX