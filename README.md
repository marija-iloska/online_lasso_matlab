# Fast Online LASSO

This code is a MATLAB implementation of our ICASSP 2025 paper by the name "Fast Sparse Learning from Streaming Data with LASSO". The proposed method is very easy to implement and we provide an example script how to run, as well as a script that reproduces the figures in the paper. 
 

> **Abstract:** In this paper, we propose Online LASSO - a version of
LASSO that is configured for streaming data. In standard LASSO,
the penalty parameter is typically chosen by cross-validation, a
procedure which requires the entire dataset upfront and repeated
fitting. The main contribution of this work is in finding an easy and
principled choice for the penalty parameter for every incoming
data point, in the cases where the input features are uncorrelated.
The proposed Online LASSO enjoys several benefits: i) it is
memory and time efficient ii) it is easy to implement, iii) it does
not require an initial batch of data to start, iv) it does not require
any tuning (e.g., step size or tolerance), and finally v) it converges
to the performance of the optimal predictor and correct selection
of features. We demonstrate these capabilities and compare Online
LASSO to standard LASSO as well as other adaptive LASSO
variations and provide discussion on their performances.
> 

# Code
## To run the proposed method <br/>
Run example_code.m - specify the system settings at the top of the script for the desired synthetic data (e.g., noise, length, sparsity). <br/>
<br/>

## To reproduce experiments in paper <br/>
Run reproduce_fig.m - specify system settings as well as parameter tuning for the competing methods at the top. To achieve statistical results, set the number of runs $R >1$ and run as parfor. <br/>
<br/>

## Organization <br/>
proposed_method/online_lasso.m  - the fn where the proposed method is implemented. <br/>

baselines/olin_lasso.m - the fn where competing method OLinLASSO is implemented. <br/>
baselines/occd.m - the fn where competing method OCCD-TWL is implemented. <br/>

util/generate_data.m - a fn that generates data. <br/>
util/metrics.m - a fn that computes f-score, mse on test data, and mse of theta. <br/>
util/stream_data.m - a fn that streams new incoming data and calls all the methods to run in order to compare. <br/>

results/ - a folder which stores the figs and .mat files of the results.


