# DimiG for inferring disease-associated miRNAs
The microRNAs (miRNAs) play crucial roles in many biological processes involved in diseases and miRNAs function with protein coding genes (PCGs). In this study, we present a semi-supervised multi-label framework to integrate PCG-PCG interactions, PCG-miRNA interactions, PCG-disease associations using graph convolutional network (GCN). DimiG is then trained on a graph, which is further used to score associations between diseases and miRNAs.

## Requirements
  * Sklearn
  * GCN
  * PyTorch 0.4 or 0.5
  * Python 2.7

## Installation of GCN
Here we modified the orginal GCN (https://github.com/tkipf/pygcn) to support multi-label learning. <br> 
```python setup.py install```
