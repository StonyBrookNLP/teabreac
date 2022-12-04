#!/bin/bash

set -e
set -x

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.
pip install gdown

ZIP_NAME="teabreac_v1.0.zip"

# URL: https://drive.google.com/file/d/1DLap7BsrwEon6vJQZdtr84Ii5rr2pt8y/view?usp=sharing
gdown 1DLap7BsrwEon6vJQZdtr84Ii5rr2pt8y --output $ZIP_NAME
unzip $(basename $ZIP_NAME)
rm $ZIP_NAME

# TODO: prevent these from zipping in.
rm -rf __MACOSX
