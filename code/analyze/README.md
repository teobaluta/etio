## ETIO graph inference and causal analysis

This folder contains scripts for generating the graph and causal analysis. It also contains other
scripts for parsing, aggregating, and analyzing the data.
We mainly present how to use the scripts pertaining to the graph inference and analysis.

### Requirements


### Graph inference

- `infer_graph.R` : Learns the graphs from a dataset of traces
Command format is the following:
```
infer_graph.R <data.csv> <output_dir> loss cont <weight_decay>
```
Inputs:
	- <data.csv> : the traces. Needs to contain the expected columns with the potential factors.
	- <output_dir>: the output directory where the graphs are saved.
	- <weight_decay>: the value of the weight decay.

Outputs:
	- Graphs: the script generates the graphs in `.dot` format and `.pdf` in two folders:
			`<output_dir>-w_dk-cont-wd_<weight_decay>` (when using the domain knowledge) and
			`<output_dir>-wo_dk-cont-wd_<weight_decay>`. The graphs for the simple hypotheses are
			generated in `<output_dir>`
	- Results: the script also generates a number of `.csv` files containing the predictive accuracy
			compared to simple correlations

#### How to Run

The following command runs inference and redirect the logs to a file:
```
Rscript infer_graph.R ../../aggregated_stats/full-summary.csv graphs loss cont 0.0005 > log-wd_0.0005-new-out.out
```
This will create `graphs-wo_dk-cont-wd_0.000500` containing the learned graphs.

The following command runs inference and redirect the logs to a file:
```
Rscript infer_graph.R ../../aggregated_stats/full-summary.csv graphs loss cont 0.005 > log-wd_0.005-new-out.out
```
This will create `graphs-wo_dk-cont-wd_0.005000` containing the learned graphs.

### Causal Analysis

Before running this, please make sure you are loading the right python environment.

#### How to Run
```
python answer_queries.py ../../aggregated_stats/full-summary.csv graphs-w_dk-cont-wd_0.000500/ --filename causal_estimates-wd_0.0005-scaler-w_dk-new-out.csv --wd 0.0005 > causal_estimates-wd_0.0005-w_dk-new-out.log
```
This will create `.csv` files corresponding to the ATE of several features of interest on the
attack.

### Pitfalls Example

To recreate and view the subset of the data used in the first example in the paper, run this script.

`python statistical_analysis.py`
