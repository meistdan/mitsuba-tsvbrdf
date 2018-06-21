#!/bin/bash

FRAMES=30

SCENE=../../scenes/plane/plane.xml

MATERIAL_PATH=d:/projects/tsvbrdf/data
OUTPUT_PATH=../../../out/plane

TSVBRDF=tsvbrdf.py
TESTS=$(ls $MATERIAL_PATH)

MATERIAL_DIR=$MATERIAL_PATH/$TEST/

CURRENT_DIR=$PWD

START=$(date)
for TEST in $TESTS; do
	OUTPUT_DIR=$OUTPUT_PATH/$TEST
	MATERIAL_DIR=$MATERIAL_PATH/$TEST
	python $TSVBRDF -f $FRAMES -i $SCENE -o $OUTPUT_DIR -M $MATERIAL_DIR
	#cp makefile $OUTPUT_DIR
	#cp clip.avs $OUTPUT_DIR
	#cp submission.vcf $OUTPUT_DIR
	#cd $OUTPUT_DIR
	#make
	#cd $CURRENT_DIR
done
END=$(date)

echo $START
echo $END
