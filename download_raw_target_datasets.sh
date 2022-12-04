#!/bin/bash

set -e
set -x

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.
pip install gdown

ZIP_NAME="raw_target_datasets.zip"

# URL: https://drive.google.com/file/d/18jLfiYkkRiJzBphNPIl-zYBAOQLCCD49/view?usp=sharing
gdown --id 18jLfiYkkRiJzBphNPIl-zYBAOQLCCD49 --output $ZIP_NAME
unzip $(basename $ZIP_NAME)
rm $ZIP_NAME

# TODO: prevent these from zipping in.
rm -rf __MACOSX