This file is the code and data source file for the paper, "Validity of feature importance for weak machine learning model".

Here is the description for each folder. Overall pipeline follows the order. 
We currently delete every result in the 4.graph folder (because of the size issue), but just by running the code inside the graph folder, you can generate all the graphs. 

To follow the flow of the paper, you can run 
1. Data generation: dataset/3.preprocessed/preprocessing.ipynb
2. Performance degradation: function/channel 1.ipynb (can be any channel)
3. Stability analysis: graph/stability.ipynb
4. importance distribution analysis: graph/importance_distribution.ipynb
5. theoretical approach: graph/theretical_analysis.ipynb


------------------------------------------
Here are the details of each folder.


1. dataset
: 1. raw 		- raw files (excel,csv) downloaded from open source 
: 2. array 	- each raw file was converted to numpy array, and generated dataset was made
: 3. preprocessed 	- each data is aligned by its feature importance (the most important feature to the first column)
		- preprocessing work is done by preprocessing.ipynb

2. function
: functions.py		-the python file which contains functions used for the experiment
: channel 1,2,3.ipynb	-the jupyter notebook file to execute the experiment

3. result
-folder to save every result in each stage of performance degradation algorithm
-in each folder, there is feature_importance.npy, feature_importance_rank.npy, performance.npy, which are that of the best performing model (before any data cut and feature cut)
-the result in each performance degradation algorithm is in data_cut and feature_cut folder
-in corr_cut, the same experiment after correlations were deleted were saved

4. graph
: stability.ipynb			-code for calculating stability and trimming the grid of performance
: theoretical_analysis.ipynb	-code to draw Figure 7
: importance_distribution.ipynb	-code to draw feature importance distribution in Section 3.3
: stability			-folder to save stability results in each dataset
: trimmed_stability		-folder to save stability results after trimming
: data_information			-folder to work on making Table 1
: feature_distribution		-folder which saves the graphs of feature importance distribution (Section 3.3)
: correlation_trend		-folder which saves the graphs for correlation cut analysis in Section 3.2
