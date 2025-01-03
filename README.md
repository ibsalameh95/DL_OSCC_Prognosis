# Multiple Instance Learning for Early Recurrence Prediction in Oral Squamous Cell Carcinoma (MILP)

This repository is the official implementation of the paper: 
**Stromal collagen for deep learning prediction of early tumor recurrence in oral cancer of squamous cell carcinoma.**

It uses the data: **[Multiple Instance Learning for Early Recurrence Prediction in Oral Squamous Cell Carcinoma](https://zenodo.org/records/10658626)**.

We developed three Multiple-Instance Learning (MIL) models for oral squamous cell carcinoma (OSCC) prognosis:
1. Two models use either **Brightfield patches** or **Confocal patches** solely to drive a prognosis for a whole-slide image (WSI) as either early relapse or late/no relapse.
2. The third model combines both image types to derive the prognosis.

## Folder Structure
```
DL_OSCC_PROGNOSIS/
├── color_maps/         # Contains color mapping files for visualization
├── prognosis_model/    # Includes model files and related scripts for prognosis
├── shell_scripts/      # Automation and utility shell scripts
├── README.md           # Documentation and project overview
└── requirements.txt    # List of Python dependencies
```

## Requirements

To install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Data

Please download the data from [Multiple Instance Learning for Early Recurrence Prediction in Oral Squamous Cell Carcinoma](https://zenodo.org/records/10658626) and move the downloaded folders into the following directory:

This dataset is the final version of the processed patches to be used for the model training and testing. More info about the dataset could be found in the dataset description.
```
project_path/Data/CroppedRegions
```

## Train and Test Prognosis Model

### Train the model
```bash
python prognosis_model/train.py
```

### Command-Line Arguments for Training

Below is a list of the command-line arguments used in training, with explanations:

- `--init_model_file` (default: ''): Path to an initial model file, if resuming or fine-tuning.
- `--model_dir` (default: `Results/prognosis_model/saved_models/`): Directory to save trained model files.
- `--slide_list_filename_train` (default: `prognosis_model/data/seg/train.txt`): File containing the list of slides for training.
- `--slide_list_filename_valid` (default: `prognosis_model/data/seg/valid.txt`): File containing the list of slides for validation.
- `--patch_size` (default: `0`): Size of the image patches.
- `--num_instances` (default: `0`): Number of instances (patches) in a bag for MIL.
- `--num_features` (default: `64`): Number of features to extract from patches.
- `--pretrained` (default: `False`): Whether to use a pretrained model (e.g., on ImageNet).
- `--num_classes` (default: `2`): Number of output classes (e.g., early relapse, late/no relapse).
- `--batch_size` (default: `1`): Number of samples per training batch.
- `--learning_rate` (default: `1e-4`): Learning rate for optimizer.
- `--weight_decay` (default: `1e-5`): Weight decay (L2 regularization) for optimizer.
- `--num_epochs` (default: `500`): Total number of training epochs.
- `--save_interval` (default: `100`): Interval (in epochs) for saving the model.
- `--metrics_dir` (default: `Results/prognosis_model/metrics/`): Directory to save training metrics (e.g., loss, accuracy).
- `--validation_dir` (default: `Results/prognosis_model/validation/`): Directory to save validation metrics.
- `--cz_imgs_list` (default: `prognosis_model/data/cz_img_list.txt`): Path to the list of confocal (CZ) image files.
- `--he_imgs_list` (default: `prognosis_model/data/he_img_list.txt`): Path to the list of HE image files.
- `--imgs_type` (default: `CZ_HE`): Specifies the type of images used (CZ, HE, or both).
- `--device_num` (default: `0`): GPU device number to use for training.

### Test the model
```bash
python prognosis_model/test.py
```

### Command-Line Arguments for Testing

Below is a list of the command-line arguments used in testing, with explanations:

- `--init_model_file` (default: `Results/prognosis_model/saved_models/model_weights__2024_08_19__17_31_46__114.pth`): Path to the trained model file to be used for testing.
- `--slide_list_filename` (default: `prognosis_model/data/seg/test.txt`): File containing the list of slides for testing.
- `--patch_size` (default: `512`): Size of the image patches.
- `--num_instances` (default: `0`): Number of instances (patches) in a bag for MIL.
- `--num_bags` (default: `100`): Number of bags to process during testing.
- `--num_features` (default: `64`): Number of features to extract from patches.
- `--num_classes` (default: `2`): Number of output classes (e.g., early relapse, late/no relapse).
- `--batch_size` (default: `1`): Number of samples per testing batch.
- `--metrics_dir` (default: `Results/prognosis_model/test_metrics/`): Directory to save testing metrics.
- `--cz_imgs_list` (default: `prognosis_model/data/cz_img_list.txt`): Path to the list of confocal (CZ) image files.
- `--he_imgs_list` (default: `prognosis_model/data/he_img_list.txt`): Path to the list of HE image files.
- `--imgs_type` (default: `CZ_HE`): Specifies the type of images used (CZ, HE, or both).

### Process Predictions
```bash
python prognosis_model/process_bag_predictions.py
```
The `process_bag_predictions.py` script processes the prediction outputs of the testing phase and generates final scores for the slide classification. This step is crucial for evaluating model performance and understanding its output.

### Command-Line Arguments for Processing Predictions

Below is a list of the command-line arguments used in processing predictions, with explanations:

- `--slide_list_filename` (default: `prognosis_model/data/seg/test.txt`): File containing the list of slides for testing.
- `--metrics_dir` (default: `Results/prognosis_model/test_metrics/2024_08_19__17_31_46__369/test`): Directory where the processed metrics and scores will be saved.

### Generate ROC Curve
```bash
python prognosis_model/slide_classification.py
```

The `slide_classification.py` script calculates Receiver Operating Characteristic (ROC) curves from the processed scores. ROC curves are essential for evaluating the diagnostic ability of the model and assessing its sensitivity and specificity across different thresholds.

### Command-Line Arguments for Generating ROC Curve

Below is a list of the command-line arguments used for ROC curve generation:

- `--metrics_file` (default: `Results/prognosis_model/test_metrics/2024_08_19__17_31_46__369/test/slide_scores.txt`): Path to the file containing processed slide scores.
- `--title` (default: `slide_scores`): Title for the ROC curve plot.


## **License**

Multiple Instance Learning for Early Recurrence Prediction in Oral Squamous Cell Carcinoma (MILP) is released under the MIT License. See the **[LICENSE](https://www.blackbox.ai/share/LICENSE)** file for details.

## **Authors and Acknowledgment**

Multiple Instance Learning for Early Recurrence Prediction in Oral Squamous Cell Carcinoma (MILP) was created by **[Ibrahim Salameh](https://github.com/ibsalameh95)**.

Additional contributors include:

- **[Mustafa Umit Oner](https://github.com/onermustafaumit)**
- **CHU, Chui Shan**
- **Lee, Nikki P**

Thank you to all the contributors for their hard work and dedication to the project.