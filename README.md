# SR-ECOC

Self-adaptively Soft-Recoding based Error Correcting Output Codes

This is the implementation for paper: [A Self-adaptively Soft-Recoding Strategy for Improving Error-Correcting Output Codes Algorithms]
## Acknowledgement

- ECOC tool kit is employed. [ECOC-TLK](<https://github.com/MLDMXM2017/ECOC_TLK>)

## Environment

- **Windows 10 64 bit**
   ```
    Single process mode
   ```
- **Windows 10 64 bit** 
   ```
    multi process mode
   ```

- **python 3.7**
  ```
    cvxopt==1.2.2
    cvxpy==1.0.15
    h5py==2.10.0
    imbalanced-learn==0.4.3
    imblearn==0.0
    ipython==7.16.1
    ipython-genutils==0.2.0
    multiprocess==0.70.7
    numpy==1.18.5
    opencv-contrib-python==4.1.2.30
    opencv-python==4.1.2.30
    preprocess-icm==0.56
    pywin32==300
    scikit-learn==0.20.2
    scipy==1.4.1
    seaborn==0.9.0
    six==1.12.0
    xgboost==1.4.2
    xlrd==1.2.0
    xlutils==2.0.0
    xlwt==1.3.0
  ```
  
## Dataset

- **Data format**
  ```data info
  Raw data is put into the folder ```($root_path)/data/```.
  There is a micro-array data set in the folder ```($root_path)/data/``` as an example. 
  Dataset information:
    name: Breast
    class num: 5
    feature num: 9216
    sample num: 84
  ```

## Runner Setup

- **Main parameters**
  ```params
  repeat = 10 # Number of run repetitions
  fold = 5    # Cross-validated fold
  alpha_list = [0.01]  # Learning rate
  Decoder = 'ED'  # Decoding scheme
  base_learner = 'SVM' 
  T = 500  # The maximum number of iterations
  ε = 0.00001  
  ecoc_types = ['dense']
   ```
- **Run the following command**
  ```python
   python SASR_test.py
  ```

## Additional Results



