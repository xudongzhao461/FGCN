# FGCN：Fractional Gabor Convolutional Network for Multi-source Remote Sensing Data Classification

This example implements the paper in review [Fractional Gabor Convolutional Network for Multi-source Remote Sensing Data Classification]

## Coming Soon. 

A Fractional Gabor Convolutional Network for Multi-source Remote Sensing Data Classification. Evaluated on the dataset of Houston, Trento and MUUFL. 

## Prerequisites
- Python 2.7 or 3.6
- Packages
```
pip install -r requirements.txt
```

## Usage

### Data set links

1. Houston dataset were introduced for the 2013 IEEE GRSS Data Fusion contest. Data set links comes from http://www.grss-ieee.org/community/technical-committees/data-fusion/2013-ieee-grss-data-fusion-contest/

2. The authors would like to thank Dr. P. Ghamisi for providing the Trento Data. 

3. The MUUFL Gulfport Hyperspectral and LIDAR Data [1][2] is Available from https://github.com/GatorSense/MUUFLGulfport/.

[1] P. Gader, A. Zare, R. Close, J. Aitken, G. Tuell, “MUUFL Gulfport Hyperspectral and LiDAR Airborne Data Set,” University of Florida, Gainesville, FL, Tech. Rep. REP-2013-570, Oct. 2013.

[2] X. Du and A. Zare, “Technical Report: Scene Label Ground Truth Map for MUUFL Gulfport Data Set,” University of Florida, Gainesville, FL, Tech. Rep. 20170417, Apr. 2017. Available: http://ufdc.ufl.edu/IR00009711/00001.

### dataset utilization

**Please modify line 10-23 in *data_util_c.py* for the dataset details.**

### Training

Train the merged HSI and LiDAR-based DSM
```
python main.py 
```

## Results
All the results are cited from original paper. More details can be found in the paper.

| dataset  	 | Kappa | OA      |
|---------- |-------  |--------|
| MUUFL    | 86.90%| 89.90% |
| Houston  | 98.38%| 98.50%|
| Trento    | 99.09%| 99.326% |
## Citation

Coming Soon


