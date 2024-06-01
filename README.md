#  DL NOVEL PROG OSCC

This research findings suggest that the ECM fibril waveforms probably serve as biomarkers for oral SCC progression. Oral cancer patients are potentially benefited from using our deep learning model for prognosis prediction to build a preparedness for cancer recurrence. Future prospective studies should examine the decision-making effects of the system for the patients.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data

> please download the data from this link https://zenodo.org/records/10658626, move downloaded folders into project_path/Data/CroppedRegions directory.


## Train and test prognosis model

```train
python prognosis_model/train.py
```

```test
python prognosis_model/test.py
```

```process predictions
python prognosis_model/process_bag_predictions.py
```


```get ROC
python prognosis_model/slide_classification.py
```

