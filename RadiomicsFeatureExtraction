# This is experimentary code

import radiomics
from radiomics import featureextractor, firstorder, glcm, imageoperations, shape, glrlm, glszm
import six

from mpl_toolkits.axes_grid1 import make_axes_locatable
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy
import seaborn as sns
import scipy.io as sio
import time

fpath = ""

stTime = time.localtime()

figPath = fpath+"PyRadiomicsFE"+str(stTime.tm_year)+"Y"+str(stTime.tm_mon)+"M"+str(stTime.tm_mday)+"D"+str(stTime.tm_hour)+"H"+str(stTime.tm_min)+str(stTime.tm_sec)+"/"
os.mkdir(figPath)

maxPatNum = 
ShapeFeaturesStorage = numpy.zeros((maxPatNum, 14))

# Enhancing Tumor + Necrosis
FeatureStorage = numpy.zeros((maxPatNum, 18 + 24 + 16))  # Hist + GLCM + ISZM + GLRLM

'''''''''''''''''''''''''''''''''''''''''''''
# Feature extractor settings
'''''''''''''''''''''''''''''''''''''''''''''
ShapeFeaturesExtractor = featureextractor.RadiomicsFeatureExtractor()
ShapeFeaturesExtractor.disableAllFeatures()
ShapeFeaturesExtractor.enableFeatureClassByName('shape')

firstOrderFeaturesExtractor = featureextractor.RadiomicsFeatureExtractor(binCount=128, verbose=True, interpolator=None)
firstOrderFeaturesExtractor.disableAllFeatures()
firstOrderFeaturesExtractor.enableFeatureClassByName('firstorder')

glcmFeaturesExtractor = featureextractor.RadiomicsFeatureExtractor(binCount=128, verbose=True, interpolator=None)
glcmFeaturesExtractor.disableAllFeatures()
glcmFeaturesExtractor.enableFeatureClassByName('glcm')

iszmFeaturesExtractor = featureextractor.RadiomicsFeatureExtractor(binCount=32, verbose=True, interpolator=None)
iszmFeaturesExtractor.disableAllFeatures()
iszmFeaturesExtractor.enableFeatureClassByName('glszm')


patNum = 1
while patNum <= maxPatNum:

    print(" ")
    print("[patNum "+str(patNum)+"] Processing Start...")
    figPathPat = figPath

    # Load images
    try:
        '''''''''''''''''''''''''''''''''''''''''''''
        # Load images
        '''''''''''''''''''''''''''''''''''''''''''''
        print("Load image files...")

        imageName = fpath + str(patNum) + ".nii.gz"
        ROIName = fpath + str(patNum) + ".roi.nii.gz"

        image = sitk.ReadImage(imageName)
        ROI = sitk.ReadImage(ROIName)
        
        ROI = (ROI != 0)
        ROI.CopyInformation(image)

        '''''''''''''''''''''''''''''''''''''''''''''
        # Find Maximum ROI Slice & save image_roi_plot
        '''''''''''''''''''''''''''''''''''''''''''''
        print("Making representative slice image ...")
        MaxSliceSize = 0
        MaxSliceNum = 0
        sliceNum = 0
        while sliceNum < ROI.GetSize()[2]:
            roiSlice = sitk.GetArrayViewFromImage(ROI)[sliceNum, :, :]
            if MaxSliceSize < numpy.count_nonzero(roiSlice):
                MaxSliceNum = sliceNum
                MaxSliceSize = numpy.count_nonzero(roiSlice)
            sliceNum += 1

        sliceNum = MaxSliceNum
        print("Representative SliceNum = " + str(sliceNum))

        roiSliceImage = sitk.GetArrayViewFromImage(image)[sliceNum, :, :]
        roiSlice = sitk.GetArrayViewFromImage(ROI)[sliceNum, :, :]

        roiSlice = numpy.ndarray.astype(roiSlice, dtype='float32')
        roiSlice = numpy.ma.masked_where(roiSlice == 0, roiSlice)

        plt.figure(figsize=(8, 4), dpi=150)
        figure = plt.subplot(1, 2, 1)
        im = figure.imshow(roiSliceImage, cmap='gray')
        plt.title('Representative Slice')

        figure = plt.subplot(1, 2, 2)
        im = figure.imshow(roiSliceImage, cmap='gray')
        my_cmap = sns.light_palette(sns.xkcd_rgb["red orange"], input="his", as_cmap=True, reverse=True)
        im2 = figure.imshow(roiSlice, cmap=my_cmap, alpha=0.4)

        plt.tight_layout()
        plt.savefig(figPathPat + "RepresentativeSlice" + str(patNum) + ".png", dpi=150, bbox_inches='tight',
                    transparent=True)
        plt.clf()
        plt.close()

        '''''''''''''''''''''''''''''''''''''''''''''
        # 3D shape based feature
        # Feature n = 14
        '''''''''''''''''''''''''''''''''''''''''''''
        print("Calculating shape features...")
        ShapeFeatures = ShapeFeaturesExtractor.execute(image, ROI)

        '''''''''''''''''''''''''''''''''''''''''''''
        # Histogram Based Features, LoG off
        # Feature n = 18
        '''''''''''''''''''''''''''''''''''''''''''''
        print("Calculating histogram features...")
        HistFeatures = firstOrderFeaturesExtractor.execute(image, ROI)

        '''''''''''''''''''''''''''''''''''''''''''''
        # GLCM Based Features, LoG OFF
        # Feature n = 24
        '''''''''''''''''''''''''''''''''''''''''''''
        print("Calculating GLCM features...")
        GLCMFeatures = glcmFeaturesExtractor.execute(image, ROI)

        '''''''''''''''''''''''''''''''''''''''''''''
        # ISZM Based Features, LoG OFF
        # Feature n = 16
        '''''''''''''''''''''''''''''''''''''''''''''
        print("Calculating ISZM features...")
        ISZMFeatures = iszmFeaturesExtractor.execute(image, ROI)

        '''''''''''''''''''''''''''''''''''''''''''''
        # Save features in storage variable
        '''''''''''''''''''''''''''''''''''''''''''''
        ShapeFeaturesStorage[patNum - 1, ] = [ShapeFeatures[x] for x in list(
            filter(lambda k: k.startswith("original_") or k.startswith("log"), ShapeFeatures))]

        FeatureStorage[patNum - 1, ] = \
            [HistFeatures[x] for x in list(filter(lambda k: k.startswith("original_"), HistFeatures))] \
            + [GLCMFeatures[x] for x in list(filter(lambda k: k.startswith("original_"), GLCMFeatures))] \
            + [ISZMFeatures[x] for x in list(filter(lambda k: k.startswith("original_"), ISZMFeatures))]


    except:
        print("[FATAL Error] Something Wrong... patNum " + str(patNum))
        patNum += 1
        continue;

    patNum += 1

# Total Feature values
TotalFeatures = numpy.concatenate((ShapeFeaturesStorage, FeatureStorage), axis=1)

# Feature names
ShapeFeatureNames = list(filter(lambda k: k.startswith("original_"), ShapeFeatures))

HistFeatureNames = list(filter(lambda k: k.startswith("original_"), HistFeatures))
GLCMFeatureNames = list(filter(lambda k: k.startswith("original_"), GLCMFeatures))
ISZMFeatureNames = list(filter(lambda k: k.startswith("original_"), ISZMFeatures))

# Pre
FeatureNames = numpy.concatenate(
    ([s for s in HistFeatureNames], [s for s in GLCMFeatureNames],
     [s for s in ISZMFeatureNames]), axis=0)

FeatureNames = numpy.concatenate((ShapeFeatureNames, FeatureNames), axis=0)

# Save result
sio.savemat(figPath+"RadiomicsFeautres.mat", {"TotalFeatures": TotalFeatures, "FeatureNames": FeatureNames})

print("")
print("Feature Extraction finished:"+time.asctime())
