import pandas as pd
import sys
import matplotlib.pyplot as plt
import sklearn.preprocessing
import numpy as np

ATTACKS = ['Oak17Acc', 'Top3MLLeakAcc', 'Top3MLLeakLAcc', 'MLLeakAcc', 'MLLeakLAcc', 'ThresholdAcc']
LOSSES = ['ce', 'mse']
FEATURES = [
    'TrainBias2', 'TestBias2',
    'TrainAcc', 'TestAcc',
    'TrainLoss', 'TestLoss',
    'TrainVar', 'TestVar',
    'LossDiff',
    'AccDiff',
    'NumParams', 'TrainSize',
    'CentroidDistance.origin.', 'CentroidDistance.sorted_3.']
TREATMENT_VARS = {
    'ThresholdAcc': [x for x in FEATURES if x not in ['CentroidDistance.sorted_3.']],
    'Oak17Acc': [x for x in FEATURES if x != 'CentroidDistance.sorted_3.'],
    'Top3MLLeakAcc': [x for x in FEATURES if x not in ['CentroidDistance.origin.']],
    'Top3MLLeakLAcc': [x for x in FEATURES if x not in ['CentroidDistance.origin.']],
    'MLLeakAcc': [x for x in FEATURES if x not in ['CentroidDistance.sorted_3.']],
    'MLLeakLAcc': [x for x in FEATURES if x not in ['CentroidDistance.sorted_3.']],
}

# For example, dataset path is ../aggregated_stats/full_summary.csv
dataset = sys.argv[1]
attack = 'Oak17Acc'
wd = 0.0005

df = pd.read_csv(dataset, skip_blank_lines=True)
df.rename(columns={'CentroidDistance(origin)': 'CentroidDistance.origin.',
                    'CentroidDistance(sorted_3)': 'CentroidDistance.sorted_3.'},
         inplace=True)
df.loc[df["TrainSize"] == "5k", "TrainSize"] = 5000.0
df.loc[df["TrainSize"] == "1k", "TrainSize"] = 1000.0
df[["TrainSize"]] = df[["TrainSize"]].apply(pd.to_numeric)
df[["NumParams"]] = df[["NumParams"]].apply(pd.to_numeric)

scaler = sklearn.preprocessing.StandardScaler()
df[TREATMENT_VARS[attack]] = scaler.fit_transform(df[TREATMENT_VARS[attack]])

# normalize the data
scaler = sklearn.preprocessing.MinMaxScaler()
df[TREATMENT_VARS[attack]] = scaler.fit_transform(df[TREATMENT_VARS[attack]])

ce_df = df.loc[(df['Loss'] == 'ce')]
ce_df = df.loc[df['WeightDecay'] == wd]
view1 = ce_df.loc[(ce_df['TrainAcc'] < 0.9) & (ce_df['TestAcc'] > 0.5)]
view2 = ce_df.loc[(ce_df['TrainAcc'] >= 0.9) & (ce_df['TestAcc'] > 0.8)]

view5 = ce_df.loc[(ce_df["AccDiff"] > 0.0) & (ce_df["AccDiff"] < 1)
                  #& (ce_df['TrainAcc'] > 0.6) #& (ce_df['TestAcc'] > 0.2)
                  ][['AccDiff','NumParams','Width','Dataset','Arch','TrainSize','WeightDecay',
                     'Oak17Acc','TrainAcc','TestAcc', 'TrainVar','Scheduler?', 'lr','EpochNum']]
cluster1 = view5.loc[(view5['NumParams'] <= 1)
                & (view5['NumParams'] > 0.02)
                & (view5['Dataset'] == 'cifar10')
                & (view5['Arch'] == 'resnet34')
                & (view5['TrainSize'] == 1.0)
                & (view5['Scheduler?'] == 'with_scheduler')
                 ]
cluster2 = view5.loc[(view5['NumParams'] <= 1)
                & (view5['NumParams'] > 0.02)
                & (view5['Dataset'] == 'cifar10')
                & (view5['Arch'] == 'densenet161')
                & (view5['TrainSize'] == 1.0)
                & (view5['Scheduler?'] == 'with_scheduler')
                 ]
# This shows that the relationship is indeed positive
# Results in Fig. 1 in the paper - for illustration purposes of the
# possible confounding effects when looking at slices of data
plt.scatter(cluster1['AccDiff'], cluster1['Oak17Acc'])
plt.scatter(cluster2['AccDiff'], cluster2['Oak17Acc'])
plt.xlabel('The Train-to-test Accuracy Gap')
plt.ylabel('Multiple Shadow Model Attack Accuracy')
plt.savefig('fig1-confounders.png')
pd.set_option('display.max_rows', None)

# Check the average treatment effect for all of the experiments for the
# multipleshadow model attack
# Computes the non-adjusted effects E1 and E3 (as shown in Fig. 2 in the paper)
# For the adjusted effects see the answer_queries.py script
pd.crosstab(df.Oak17Acc, df.AccDiff, margins=True, normalize='columns')['All'].sum()
max_accdiff = max(df.AccDiff)
min_accdiff = min(df.AccDiff)
maxdiff_cond = df.loc[(np.abs(df['AccDiff'] - max_accdiff) < 0.001)]
mindiff_cond = df.loc[(np.abs(df['AccDiff'] - min_accdiff) < 0.001)]
cond_avg = maxdiff_cond['Oak17Acc'].mean() - mindiff_cond['Oak17Acc'].mean()
print('Estimated conditional expectation E[Attack|AccDiff=max] - ' \
      'E[Attack|AccDiff=min] = {} (E1)'.format(cond_avg))
print('\tE[Attack|AccDiff=max] = {} ; '\
      'E[Attack|AccDiff=min] = {}'.format(maxdiff_cond['Oak17Acc'].mean(),
                                          mindiff_cond['Oak17Acc'].mean()))

max_numparams = df['NumParams'].max()
min_numparams = df['NumParams'].min()
mindiff_cond = df.loc[(np.abs(df['NumParams'] - min_numparams) < 0.001)]
maxdiff_cond = df.loc[(np.abs(df['NumParams'] - max_numparams) < 0.001)]
cond_avg = maxdiff_cond['Oak17Acc'].mean() - mindiff_cond['Oak17Acc'].mean()
print('Estimated conditional expectation E[Attack|NumParams=max] - '
      'E[Attack|NumParams=min] = {} (E3)'.format(cond_avg))
print('\tE[Attack|NumParams=max] = {} ; ' \
      'E[Attack|NumParams=min] = {}'.format(maxdiff_cond['Oak17Acc'].mean(),
                                            mindiff_cond['Oak17Acc'].mean()))