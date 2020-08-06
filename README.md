# LN Metastasis Prediction using Deep Learning

Lymph node metastasis prediction from ultrasound images using deep learning

If you use this code in your research, consider citing:
```
@article{
  title={xxxx},
  author={xxxx},
  journal={xxxx},
  year={xxxx},
  publisher={xxxx}
}
```

## Prerequisites

- Ubuntu 18.04 with Nivida 2080Ti
- Python 3.6 with dependencies listed in the `requirements.txt` file
```Bash
   sudo pip install -r requirements.txt
```

## Running

1. clone the repo to local directory
```Bash
   git clone https://github.com/MedicalDataAI/LNMP.git
```

2. download the weight file of the trained model into the folder of "./models"
```Bash
   wget 'https://drive.google.com/file/d/1Y3wGUpS_wQAA5szQDnkuBU2GhJftN6Qv/view?usp=sharing'
   wget 'https://drive.google.com/file/d/15e2z8qL2r7-zHLRF-jydMGwFr9lLrjtF/view?usp=sharing'
   wget 'https://drive.google.com/file/d/1lItMkSpHfpiXA8XPVjV1jOF1ss0HtBAj/view?usp=sharing'
   wget 'https://drive.google.com/file/d/1AlUMmmKb53yvUQMzRZT1ikibLekSu9HK/view?usp=sharing'
```

4. use the trained model to predict the data of Clinical, BMUS, CDFI (NOTE: to replace the parameter of path with proper location of model file)

- Predict clinical data (NOTE: to replace the data in "./data/clinical.csv") into "./res/result_clinical.csv"
```Bash
   python3 clinical_lr.py
```   

- Predict BMUS data (NOTE: to replace the images in "./data/bmus/*") into "./res/result_bmus.csv"
```Bash
   python3 bmus_cnn.py
```

- Predict CDFI data (NOTE: to replace the images in "./data/cdfi/*") into "./res/result_cdfi.csv"
```Bash
   python3 cdfi_cnn.py
```

- Predict ensemble data (Prompt for input the risk of Clinical, BMUS and CDFI from clinical_lr.py, bmus_cnn.py and cdfi_cnn.py.)
```Bash
   python3 ensemble_bagging.py
```

