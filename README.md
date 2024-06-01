#  DL NOVEL PROG OSCC

Reproducing the results of the paper: Artificial intelligence quantified tumour-stroma ratio is an independent predictor for overall survival in resectable colorectal cancer
Zhao, K., Li, Z., Yao, S., Wang, Y., Wu, X., Xu, Z., ... & Liu, Z. (2020). Artificial intelligence quantified tumour-stroma ratio is an independent predictor for overall survival in resectable colorectal cancer. EBioMedicine, 61, 103054.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data

> please download the data from this link https://zenodo.org/record/4024676#.Y49odS8RpQI, move downloaded folders into project_path/data/ directory.

## Segmentation model
 
* sh ShellScripts/6_train_test_model.sh
	* Takes train and validation datasets, model parameters and start model training and validation. 
	* Data visualization for the training and validation results.


## Diagnosis model


```train
shell_scripts/10_train_test_prognosis_model2.sh
```


## Prognosis model

```train
shell_scripts/13_train_prognosis_model3.sh
```