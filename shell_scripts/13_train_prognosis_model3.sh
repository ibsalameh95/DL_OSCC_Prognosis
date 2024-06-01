echo 'Train prognosis model phase 3'

python prognosis_model/train.py --num_features 64 --image_dir Data/CroppedRegions/512/ExtractedFeatures/CZ --slide_list_filename_train prognosis_model/Data/train.txt --slide_list_filename_valid prognosis_model/Data/valid.txt