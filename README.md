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
- **Accuracy Metric Result**

![image](https://github.com/MLDMXM2017/SA-soft-recoding/blob/main/A1.jpg)
![image](https://github.com/MLDMXM2017/SA-soft-recoding/blob/main/A2.jpg)

- **Comparison of Computational Cost**

![image](https://github.com/MLDMXM2017/SA-soft-recoding/blob/main/A3.jpg)
   Table A3 shows the computational time for all the data sets, including training (denoted as Tr) and test (denoted as Te) times, in the experimental environment: Linux Ubuntu-16.04 operating system, Intel(R) Xeon(R) E5-2665 CPU, 120GB memory size. It can be seen that, due to the introduction of the iteratively update process in the soft-recoding phase, our method requires longer time to train compared to those of the HC schemes, but it requires almost the same test time as the original ECOC algorithms. As the system is usually trained offline, the longer training time is generally acceptable as long as it has higher performance along with a similar classification time.
   The numbers of samples and classes are the main factors affecting the computational time of a model. For example, $Led24digit$ has a relatively large number of data samples (1000 samples) and class numbers (10 classes), resulting in a significant increase in time consumption. In addition, the convergence speed is also a key factor affecting the calculation time. With the increase of the iteration numbers, the computational time increases accordingly. A typical example is that although D-ECOC has fewer columns compared to OVA, when applied to the $Gcm$ data set, the average number of epochs for the former is 179.2, while that for the latter is 64. Therefore, the former runs longer than the latter (33.69s versus 16.1s).

- **Analysis of Significant Level in Performance**

![image](https://github.com/MLDMXM2017/SA-soft-recoding/blob/main/A4.jpg)
   The Friedman test and the Nemenyi test are employed to verify whether the self-adaptive algorithm is significantly different from other algorithms. Friedman test is a non-parametric statistical test. It employs performance ranks of all algorithms for tests. Let $k$ and $D$ denote the number of algorithms and the number of data sets, respectively, $r_j^i$ denotes the rank of $j$-th algorithm on the $i$-th data set and $r_j$ represents the average rank on the $j$-th algorithm, as shown in \cref{eq:15}. Then the Friedman statistic for each metric is calculated by \cref{eq:16}. Based on this equation, Iman and Davenport proposed an improved version in \cref{eq:17}. 
![image](https://github.com/MLDMXM2017/SA-soft-recoding/blob/main/A5.jpg)
![image](https://github.com/MLDMXM2017/SA-soft-recoding/blob/main/A6.jpg)


