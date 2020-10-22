# Implementations for: Overcoming The Limitations of Neural Networks in Composite-Pattern Learning with Architopes
**Submitted to:** *The 24th International Conference on Artificial Intelligence and Statistics*

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

The first trains the semi-supervised architope model and the following benchmark modelds: ffNN, GBRF, ffNN-bag, and ffNN-lgt.  The subsequent two scripts, respectively, train the architope on the expert-provided partition described [here](https://github.com/ageron/handson-ml/tree/master/datasets/housing.) and architope obtained from an additional repartitioning step, as described by Algorithm 3.2 of the paper.  

---

## Results

Our models and their benchmarks achieves the following performance on:

### [California Housing Price Dataset](https://github.com/ageron/handson-ml/tree/master/datasets/housing)

The house prices were multiplied by $10^{-5}$ to avoid exploding gradient issues.

1. For Testing:

|Model |  MAE |       MSE |       MAPE |
|-|-|-|-|
| ffNN     |  0.321383 |  0.251979 |  19.810062 |
| GBRF     |  0.346361 |  0.259093 |  17.985115 |
| ffNN-bag |  0.495927 |  0.461367 |  29.052931 |
| ffNN-lgt |  0.318711 |  0.257818 |  19.901575 |
| Architope     |  0.312752 |  0.249929 |  17.689563 |


---
Hyperparameter Grid Used in Training for the paper ["Non-Euclidean Universal Approximation"](https://arxiv.org/abs/2006.02341)

| Batch size | Epochs | Learning Rate | Height (Middle Layers) | Depth - Input Layers | Depth - Middle Layers | Depth - Output Layers |
|------------|--------|---------------|------------------------|----------------------|-----------------------|-----------------------|
|     16     |  200   |    0.0001     |         200            |          2           |           1           |            2          |
|      32    |  400   |    0.0005     |         250            |          3           |           2           |            3          |
|     -      |  800   |    0.005      |         400            |          4           |           -           |            4          |
|     -      |  1000  |      -        |         600            |          5           |           -           |            5          |
|     -      |  1200  |      -        |         800            |          -           |           -           |            -          |
|     -      |  -     |      -        |        1000            |          -           |           -           |            -          |


### Meta-parameters Used:
- n_jobs = 70 (Number of cores used in training).
- n_iter = 10 (The number of cross-validation iterations used/per model when performing the grid-search).
- CV_folds = 4 (The number of cross-validation folds used).
