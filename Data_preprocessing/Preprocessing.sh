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
	#input_image="$image.nii"
	LPI_image="LPI_$image.nii"
	BFcor_image="BFcor_$image.nii"
	mask_image="mask_$image.nii"
	SS_image="SS_$image.nii"
	image_MNI="[MNI]$image.nii"
	MNI_mat="[MNI]$imageI.mat"
	
	beginTime=$(date +%s%N)
	for i in {1..5}
	do
		echo "stage ($i) step"
		case $i in
			1)
			echo "Deoblique"
			3drefit -deoblique $entry
			;;
			2)
			echo "LPI"
			3dresample -orient LPI -prefix $LPI_image -inset $entry
			;;
			3)
			echo "BFC"
			3dUnifize -input $LPI_image -prefix $BFcor_image -T2 -clfrac 0.3 -T2up 99.5
			;;
			4)
			echo "3dSkullStrip"		
			(3dSkullStrip -mask_vol -input $BFcor_image  -prefix $mask_image)
			;;
			5)
			echo "3dcalc"
			3dcalc -a $BFcor_image -b $mask_image -expr 'a*step(b)' -prefix $SS_image
			;;
		esac
	done

	endTime=$(date +%s%N)
	elapsed=`echo "($endTime - $beginTime) / 1000000"`
	elapsedSec=`echo "scale=6;$elapsed / 1000"  | awk '{printf "%.6f", $1}'`
	echo TOTAL: $elapsedSec sec
done

python zslack.py --experiment SkullStirp
