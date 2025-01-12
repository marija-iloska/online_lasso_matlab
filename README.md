# Online LASSO

Note: A Python implementation can be found here (in progress).


This code is a MATLAB implementation of our ICASSP 2025 paper by the name "Fast Sparse Learning from Streaming Data with LASSO". The proposed method is very easy to implement and we provide an example script how to run, as well as a script that reproduces the figures in the paper. We remark that the proposed method converges only for uncorrelated features.


## Code
### To run the proposed method <br/>
Run example_code.m - specify the system settings at the top of the script for the desired synthetic data (e.g., noise, length, sparsity). <br/>
<br/>

### To reproduce experiments in paper <br/>
Run reproduce_fig.m - specify system settings as well as parameter tuning for the competing methods at the top. To achieve statistical results, set the number of runs $R >1$ and run as parfor. <br/>
<br/>

### Organization <br/>
proposed_method/online_lasso.m  - the fn where the proposed method is implemented. <br/>

baselines/olin_lasso.m - the fn where competing method OLinLASSO is implemented. <br/>
baselines/occd.m - the fn where competing method OCCD-TWL is implemented. <br/>

util/generate_data.m - a fn that generates data. <br/>
util/metrics.m - a fn that computes f-score, mse on test data, and mse of theta. <br/>
util/stream_data.m - a fn that streams new incoming data and calls all the methods to run in order to compare. <br/>

results/ - a folder which stores the figs and .mat files of the results.


