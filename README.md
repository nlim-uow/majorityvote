# A Diversity-aware Model for Majority Vote Ensemble Accuracy

## Abstract
Ensemble classifiers are a successful and popular approach for classification, and are frequently found to have better generalization performance than single models in practice. Although it is widely recognized that `diversity' between ensemble members is important in achieving these performance gains, for classification ensembles it is not widely understood which diversity measures are most predictive of ensemble performance, nor how large an ensemble should be for a particular application. In this paper, we explore the predictive power of several common diversity measures and show -- with extensive experiments -- that contrary to earlier work that finds no clear link between these diversity measures (in isolation) and ensemble accuracy instead by using the $\rho$ diversity measure of Sneath and Sokal as an estimator for the dispersion parameter of a Polya-Eggenberger distribution we can predict, independently of the choice of base classifier family, the accuracy of a majority vote classifier ensemble ridiculously well. We discuss our model and some implications of our findings -- such as diversity-aware (non-greedy) pruning of a majority-voting ensemble.

## Contents of the repository
This repository contains the Matlab code used to validate the Polya-Eggenberger model on the real-world datasets as described in the paper below

## Cite as
If this research is helpful, please consider citing this research as 

     @inproceedings{DBLP:conf/aistats/LimDiversityAware2020,
     author    = {Nick J.S, Lim and Robert J, Durrant},
     title     = {A Diversity-aware Model for Majority Vote Ensemble Accuracy},
     booktitle = {Proceedings of the 23th International Conference on Artificial Intelligence
                 and Statistics, {AISTATS} 2020},
     series    = {Proceedings of Machine Learning Research},
     year      = {2020}
     }
