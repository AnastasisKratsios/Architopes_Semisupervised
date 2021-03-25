# Implementations for: Overcoming The Limitations of Neural Networks in Composite-Pattern Learning with Architopes
**Submitted to:** *Thirty-eighth International Conference on Machine Learning*

---
![alt text](https://raw.githubusercontent.com/AnastasisKratsios/Architopes_Semisupervised/main/DEMO.png)

#### DEMO: 
* In-sample approximation of discontinuous function f with discontinuities arising from two sub-patterns: f_1 and f_2 defined on a partition of the input space.  NB the (feed-forward neural network) ffNN cannot capture the sub-pattern behaviour while the architope (tope) can.
---

## Requirements

To install requirements:
*  Install [Anaconda](https://www.anaconda.com/products/individual)  version 4.8.2.
* Create Conda Environment
``` pyhton
# cd into the same directory as this README file.

conda create python=3.8 --name architopes \
conda activate architopes \
pip install -r requirements.txt
```
---

## Organization of directory:
 - Data in the "inputs" sub-directory,
 - All model outputs go to the "outputs" subdirectory,
 - Jupyternotebook versions of the python files and auxiliary "standalone python scripts" are found in the "Auxiliary_Codes" subdirectory.  

---

## Preprocessing, Training, and Evaluating
1. Specify the parameters related to each set and the space of hyper parameters in the "Grid_Enhanced_Network.py" script.   

2. Preprocessing data, train models and obtaining predictions can all be done by executing the following commands:
python3.8 Architope.py
python3.8 Architope_Expert.py
python3.8 Architope_Expert_Semi_Supervised_Standalone.py

The first trains the semi-supervised architope model and the following benchmark modelds: ffNN, GBRF, ffNN-bag, and ffNN-lgt.  The subsequent two scripts, respectively, train the architope on the expert-provided partition described [here](https://github.com/ageron/handson-ml/tree/master/datasets/housing) and architope obtained from an additional repartitioning step, as described by Algorithm 3.2 of the paper.  

---

## Experimental Results

### 1) Tracking Portfolio of Day-Ahead Bitcoin Price
The trajectory of a typical cryptocurrency's price process is best modeled by a function with many jump discontinuities.  This makes predicting the day-ahead bitcoin's price based on the current day's returns of 15 *(other)* major cryptocurrencies's returns a natural testing ground for the architope.  Our result


#### Test-set performance:

|Model |  MAE |       MSE |       MAPE |
|-|-|-|-|
| ffNN     |  104.624726 |  16953.296244 |  126.453045 |
| GBRF     |  83.961955 |  11904.461155 |  177.761026 |
| ffNN-bag |  91.328308 |  13419.106120 |  185.029975 |
| ffNN-lgt |  85.132590 |  10892.687762 | 154.189022 |
| - | -| -| -| -|
| Architope     |  82.918249 |   9020.612828 |  268.814960 |


---
#### Model Complexity Results:

| In-Line (L-Time) | Parallel (P-Time) |    Number of Parameters Required |      AIC-like |    Eff(†) |
| - | -| -| -| -|
| Vanilla ffNN       |          675.123 |                 - |    52601 |   105192.699 |  1137.322 |
| Grad.Bstd Rand.F   |            34.772 |                 - |  1002480 |  2004951.139 |  1160.185 |
| Bagged ffNN        |          1078.77 |           576.093 |    25833 |    51656.971 |   927.842 |
| Architope-logistic |          1081.18 |           578.502 |    25891 |    51773.112 |   865.088 |
| - | -| -| -| -|
| Architope          |          1604.63 |           1101.95 |    42232 |    84455.164 |   883.157|

*(†) Eff is a non-standard metric, not included in the final paper.  It is defined by N_parameters x log(Test-set-MAE)*

---

![alt text](https://raw.githubusercontent.com/AnastasisKratsios/Architopes_Semisupervised/main/0_Experimental_Results/crypto/outputs/plotsANDfigures/CryptoMarketReturns.png)

---

### 2) California Housing Market
The [California Housing Price Dataset](https://github.com/ageron/handson-ml/tree/master/datasets/housing) is a dataset where an expert provided partition of the input space, based on geo-special features, is exogenously known.  This makes this dataset a good candidate for comparing our training Algorithm's ability to infer a good partition to a known "good partition".  Our models and their benchmarks achieves the following performance on:


*NB:* The house prices were multiplied by $10^{-5}$ to avoid exploding gradient issues.

#### Test-set performance:

|Model |  MAE |       MSE |       MAPE |
|-|-|-|-|
| ffNN     |  0.321383 |  0.251979 |  19.810062 |
| GBRF     |  0.346361 |  0.259093 |  17.985115 |
| ffNN-bag |  0.495927 |  0.461367 |  29.052931 |
| ffNN-lgt |  0.318711 |  0.257818 |  19.901575 |
| - | -| -| -| -|
| Architope     |  0.312752 |  0.249929 |  17.689563 |
| Architope Expert | 0.317560 | 0.256397 | 17.917502 |
| Architope Expert + Repartitioning | 0.317194 | 0.257047 | 17.508693 |


---
#### Model Complexity Results:

| In-Line (L-Time) | Parallel (P-Time) |    Number of Parameters Required |      AIC-like |    Eff(†) |
| - | -| -| -| -|
| Vanilla ffNN       |          9284.53 |                 - |    370801 |  7.416043e+05 |  4.121 |
| Grad.Bstd Rand.F   |            59.23 |                 - |  17729040 |  3.545808e+07 |  5.781 |
| Bagged ffNN        |          6361.83 |           2886.82 |     28250 |  5.650140e+04 |  5.083 |
| Architope-logistic |          6371.71 |           2896.71 |     28324 |  5.665029e+04 |  3.267 |
| - | -| -| -| -|
| Architope          |          12757.4 |           9282.36 |     30349 |  6.070032e+04 |  3.228 |
| Architope-Expert | 8453.926916 |  4117.519696 |          83731 |  167464.294 |  3.6 |
| Architope-Expert + Repartitioning |  15811.70576 |  6596.203372 |          13604 |  27210.296 |  3.019 |

*(†) Eff is a non-standard metric, not included in the final paper.  It is defined by N_parameters x log(Test-set-MAE)*

---

### 3) Synthetic Univariate Example
Our last example allows us to visually scrutinize the difference between the feed-forward architecture and the architope.  Specifically, we qualitatively and quantitatively examine our architecture's ability (and its training algorithm) to learn patterns with jump discontinuities.  We turn to a univariate illustration to streamline our visualization of this phenomenon.  


#### Test-set performance:

|Model |  MAE |       MSE |       MAPE |
|-|-|-|-|
| ffNN     |  0.041288 |  0.008275 |  63.413685  |
| GBRF     |  0.105796 |  0.014667 |  37.346306 |
| ffNN-bag |  0.060555 |  0.006006 |  89.023286 |
| ffNN-lgt |  0.054622 |  0.009862 |  40.509927 |
| - | -| -| -| -|
| Architope     |  0.015508 |  0.001026 |  17.543040 |


---
#### Model Complexity Results:

| In-Line (L-Time) | Parallel (P-Time) |    Number of Parameters Required |      AIC-like |    Eff(†) |
| - | -| -| -| -|
| Vanilla ffNN       |          1901.83 |                 - |   753001 |  1506008.374 |  0.559 |
| Grad.Bstd Rand.F   |            1.042 |                 - |       70 |      144.492 |  0.449 |
| Bagged ffNN        |          625.932 |           320.119 |   530250 |  1060505.608 |  0.798 |
| Architope-logistic |          629.353 |            323.54 |   530253 |  1060511.815 |  0.720  |
| - | -| -| -| -|
| Architope          |          2295.46 |           1989.65 |  1284749 |  2569506.333 |  0.218|

*(†) Eff is a non-standard metric, not included in the final paper.  It is defined by N_parameters x log(Test-set-MAE)*



---

![alt text](https://raw.githubusercontent.com/AnastasisKratsios/Architopes_Semisupervised/main/DEMO.png)

---
## Grid of Hyperparameters

### For feed-forward models/sub-models
|Epochs | Batch Size |  Learning Rate | Height | Depth |
| - | - | - | - | - |
| 200 |	16 | 0.0001 |    100 | 1|
| 400 |	32 | 0.0005 |    200  | 2
| 800 |	64 | 0.005 |     400  | -|
| 1000 |	- | - |    600 | -|
| 1200 |	- | - |    800 | -|
| 1400 | - | - | - | - |
		
### For deep classifiers (sub-models)
| Epochs | Batch Size | Learning Rate | Height | Depth |
| - | - | - | - | - |
| 200 |	16 | 0.0001 |    100 | 1|
| 400 |	32 | 0.0005 |    200  | 2|
| 800 |	64 | 0.005 |     400  | 3|
| 1000 |	- | 0.01 |    500 | 4|
| 1200 |	- | - |    600 | -|
| 1400 | - | - | - | -|

### For GBRF model
|Maximum Depth | Minimum Sample Leaves | Learning Rate | Number of Estimators |
| - | -| -| -|
|	1 |	1 | 0.0001 |    0.0001 | 5|
|	2 |	2 | 0.0005 |    0.0005 | 10 |
|	3 |	3 | 0.005 |     0.005 | 25 |
|	4 |	4 | 0.01 |    0.01 | 50|
|	5 |	5 | - |  - | 100 |
|	7 | 9 | - | - | 200 |
|	10 | 17 |  - | 250 |
|	25 | 20 |  - | - |
|	50 | 50 | - | - |
|	75 | 75 | - | - |
|	100 | 100 | - | - |
|		200 | - | - | - |
|	300 | - | - | - |
|	500 | - | - | - |
| - | -|-|-|
 
#### Meta-parameters Used in Cross-Validation:
- n_jobs = 70 (Number of cores used in training).
- n_iter = 50 (The number of cross-validation iterations used/per model when performing the grid-search).
- n_folds = 4 (The number of cross-validation folds used). 
