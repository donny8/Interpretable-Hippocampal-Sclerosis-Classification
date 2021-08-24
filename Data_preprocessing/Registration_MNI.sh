#!/bin/bash
searchdir=./data
cnt=0
for entry in $searchdir/*
do
	cnt=$((cnt+1))
	image=${entry:7:12}
	echo $image
done
	
echo "!!!Start PreProcesing HS Image total : [$cnt] !!!"
echo ""

for entry in $searchdir/*
do
	image=${entry:7:12}
	echo $image
	SS_image="SS_$image.nii"
	image_MNI="[MNI]$image.nii"
	MNI_mat="[MNI]$imageI.mat"
	
	beginTime=$(date +%s%N)
	for i in {1..2}
	do
		echo "stage ($i) step"
		case $i in
			1)
			echo "MNI"
            echo $SS_image
			flirt -in $SS_image -ref zMNI_1x1x1_brain.nii.gz -out $image_MNI -omat $MNI_mat -dof 12
			;;
		esac
	done

	endTime=$(date +%s%N)
	elapsed=`echo "($endTime - $beginTime) / 1000000"`
	elapsedSec=`echo "scale=6;$elapsed / 1000"  | awk '{printf "%.6f", $1}'`
	echo TOTAL: $elapsedSec sec
done

python zslack.py --experiment MNI_1_origin

