import os
import sys
import nibabel as nib
import gzip
import glob
import imp
#import scipy.misc
import numpy as np

inputX = []
listFileName = []
dirBeforeSet = os.getcwd()+"/data/before_resize"
dirAfterSet = os.getcwd()+"/data/after_resize"

if(not(os.path.isdir(dirBeforeSet))):os.mkdir(dirBeforeSet)
if(not(os.path.isdir(dirAfterSet))):os.mkdir(dirAfterSet)

image_dir = dirBeforeSet
for (imagePath, dir, files) in sorted(os.walk(image_dir)):
    if(imagePath == image_dir):
        # image reading #
        files = sorted(glob.glob(imagePath + "/*.nii.gz"))
        listFileName += files

    for i, fname in enumerate(files):
        print(fname)
        if(i==0):
            niiInfo = nib.load(fname)
        img = nib.load(fname).get_data()
        inputX.append(img)
        
inputX = np.array(inputX)
imgCount = len(inputX)
for loopCnt in range(imgCount): 
        
    npImg = np.array(inputX[loopCnt])
    tmpImg = npImg[20:180,20:220,0:170] #resize MNI space 160x200x170(before 197x233x189)
    ni_img = nib.Nifti1Image(tmpImg, np.eye(4), niiInfo.header)
    x,y = listFileName[loopCnt].split('[MNI]')
    a,b = y.split('.nii')
    print(a)
    SaveFileName = dirAfterSet+'/(resize)%s.nii' %(a)
    GzSaveFileName = '%s.gz' %(SaveFileName)

    nib.save(ni_img, SaveFileName)
    # Open output file.
    with open(SaveFileName, "rb") as file_in:
        # Write output.
        with gzip.open(GzSaveFileName, "wb") as file_out:        
            file_out.writelines(file_in)
    if os.path.isfile(SaveFileName):
        os.remove(SaveFileName)
        
    del npImg
    del tmpImg
    del ni_img
