# DimiG for inferring disease-associated miRNAs
The microRNAs (miRNAs) play crucial roles in many biological processes involved in diseases and miRNAs function with protein coding genes (PCGs). In this study, we present a semi-supervised multi-label framework to integrate PCG-PCG interactions, PCG-miRNA interactions, PCG-disease associations using graph convolutional network (GCN). DimiG is then trained on a graph, which is further used to score associations between diseases and miRNAs.

# System Requirements

## Hardware Requirements

The DimiG requires only a standard computer with enough RAM to support the operations. For minimal performance, this will be a computer with about 8 GB of RAM. For optimal performance, we recommend a computer with the following specs:

  * RAM: 8+ GB 
  * CPU: Intel® Core™ i5-3337U CPU @ 1.80GHz × 4

### OS Requirements

This package is supported for *Linux* operating systems. The package has been tested on the following systems:

Linux: Ubuntu 16.04  


## Package dependencies
  * <a href=https://github.com/scikit-learn/scikit-learn>sklearn</a> <br>
  * GCN
  * <a href=https://pytorch.org/>PyTorch 0.4 or 0.5</a> <br>
  * Python 2.7

## Installation of GCN
Here we modified the orginal GCN (https://github.com/tkipf/pygcn) to support multi-label learning. <br> 
```python setup.py install```

# Demo
1. To run the demo code, some big file needs be downloaded from other website: <br>
   - "9606.protein.links.v10.txt.gz" can be downloaded from <a href="https://string-db.org/">STRING</a> v10 database. <br>
   - "human_disease_integrated_full.tsv" can be downloaded from <a href="https://diseases.jensenlab.org/Downloads">DISEASES </a> database, it should be noted it is biweekly updated, maybe a little different from the data we used. <br>
   - "9606.v1.combined.tsv.gz" can be downloaded from <a href="https://rth.dk/resources/rain/">RAIN</a> v1.0 database. <br>
   - The above two files need be saved at dir "data/". <br> 

2. You can directly  get the prediction and give the ROC curve by running: <br>
``` run python DimiG.py``` <br>

 It takes < 30 minutes to run.
