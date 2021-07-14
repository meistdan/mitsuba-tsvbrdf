#!/bin/bash

FRAMES=21
MITSUBA_EXE='../../../build/binaries/Release/mitsuba.exe'
TSVBRDF_PY='../../scripts/tsvbdrf/tsvbrdf.py'
SCENE_FILE=scene.xml
OUTPUT_DIR=images
MATERIAL_DIR=tsvbrdf

START=$(date)
echo $PWD
python $TSVBRDF_PY -m $MITSUBA_EXE -f $FRAMES -i $SCENE_FILE -o $OUTPUT_DIR -M $MATERIAL_DIR
END=$(date)

echo $START
echo $END
