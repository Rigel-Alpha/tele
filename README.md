# Code Framework

This project conducts experiments on the **Tele dataset** and **SiChuan dataset**. Due to privacy concerns related to the Tele dataset, only the processing steps for the **SiChuan dataset** are publicly available.

---

## **Feature Preprocessing**
All processing steps applied to the raw data, including feature extraction and feature selection, are documented in `SiChuanPreprocess.ipynb` and `SiChuanPreprocess_avg.ipynb`.  
- Running the first script (`SiChuanPreprocess.ipynb`) will generate a **feature-engineered dataset** using MLFE and save the necessary files for subsequent training in `data/SiChuan/`.
- Running the second script (`SiChuanPreprocess_avg.ipynb`) will generate a **feature-engineered dataset** using a simple averaging method and save the required files in `data/SiChuan_avg/`.

---

## **Training Models**
The `TabBench` framework is used for model training and includes all the methods tested in this study.  
To reproduce our results, set the dataset (e.g., SiChuan or SiChuan_avg) and run `train.sh`. This script will:
- Train and perform hyperparameter tuning on the specified dataset using traditional or deep learning models.
- Generate prediction results (`submit.csv`) and store them in the `data/{dataset}` folder.
- Create `submit_observe.csv`, which includes the low-risk prediction probability for each user.

---

## **Environment Setup**
All dependencies are listed in `requirements.txt` and can be installed using the following command:
```bash
pip install -r requirements.txt
