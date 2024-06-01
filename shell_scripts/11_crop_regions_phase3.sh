# echo "Train and test tumor non-tumor model"

# echo 'Crop regions from confocal CZI images:'

# python prognosis_model/dataset_preperation/crop_czi_tiles.py


echo 'Crop regions from brightfield H&E images:'

python prognosis_model/dataset_preperation/crop_he_tiles.py

echo 'SegmentImages:'

python segmentation_model/tile_segmentation.py --evalimage_dir Data/CroppedRegions/Level1/1024/Brightfield --out_dir Data/CroppedRegions/Level1/1024/BrightfieldSegmented/ --stride 7


# echo 'Crop regions from the segmentation maps:'

# python prognosis_model/dataset_preperation/crop_SegMap.py