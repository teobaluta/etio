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

# scaler = sklearn.preprocessing.StandardScaler()
# df[TREATMENT_VARS[attack]] = scaler.fit_transform(df[TREATMENT_VARS[attack]])

# # normalize the data
# scaler = sklearn.preprocessing.MinMaxScaler()
# df[TREATMENT_VARS[attack]] = scaler.fit_transform(df[TREATMENT_VARS[attack]])

plt.scatter(df['AccDiff'], df['Oak17Acc'])
plt.savefig('all_points.png')
ce_df = df.loc[(df['Loss'] == 'ce')]

df = df.loc[df['WeightDecay'] == wd]
view1 = ce_df.loc[(ce_df['TrainAcc'] < 0.9) & (ce_df['TestAcc'] > 0.5)]
view2 = ce_df.loc[(ce_df['TrainAcc'] >= 0.9) & (ce_df['TestAcc'] > 0.8)]

view5 = ce_df.loc[(ce_df["AccDiff"] > 0.2) & (ce_df["AccDiff"] < 1)
                  #& (ce_df['TrainAcc'] > 0.6) #& (ce_df['TestAcc'] > 0.2)
                  ][['AccDiff','NumParams','Width','Dataset','Arch','TrainSize','WeightDecay',
                     'Oak17Acc','TrainAcc','TestAcc', 'Scheduler?', 'lr','EpochNum']]
cluster1 = view5.loc[(view5['NumParams'] <= 1)
                & (view5['NumParams'] > 0.02)
                & (view5['Width'] < 64)
                & (view5['AccDiff'] < 0.5) # for illustration purposes
                 ]
cluster2 = view5.loc[(view5['NumParams'] <= 1)
                & (view5['NumParams'] > 0.02)
                & (view5['Width'] > 64)
                & (view5['Width'] <= 256) #
                & (view5['AccDiff'] < 0.5) # for illustration purposes
                 ]
cluster3 = view5.loc[(view5['NumParams'] <= 0.02)
                    & (view5['NumParams'] > 0.01)
                    & (view5['AccDiff'] > 0.5) # for illustration purposes
                    ]

plt.scatter(cluster1['AccDiff'], cluster1['Oak17Acc'])
plt.scatter(cluster2['AccDiff'], cluster2['Oak17Acc'])
plt.scatter(cluster3['AccDiff'], cluster3['Oak17Acc'])
plt.savefig('fig1-confounders.png')
pd.set_option('display.max_rows', None)

# Check the average treatment effect
pd.crosstab(df.Oak17Acc, df.AccDiff, margins=True, normalize='columns')['All'].sum()
max_accdiff = max(df.AccDiff)
min_accdiff = min(df.AccDiff)
maxdiff_cond = df.loc[(np.abs(df['AccDiff'] - max_accdiff) < 0.001)]
mindiff_cond = df.loc[(np.abs(df['AccDiff'] - min_accdiff) < 0.001)]
cond_avg = maxdiff_cond['Oak17Acc'].mean() - mindiff_cond['Oak17Acc'].mean()
print('Estimated conditional expectation E[Attack|AccDiff=min->max] {}'.format(cond_avg))

max_numparams = df['NumParams'].max()
min_numparams = df['NumParams'].min()
mindiff_cond = df.loc[(np.abs(df['NumParams'] - min_numparams) < 0.001)]
maxdiff_cond = df.loc[(np.abs(df['NumParams'] - max_numparams) < 0.001)]
mindiff_cond['Oak17Acc'].mean()
maxdiff_cond['Oak17Acc'].mean()
cond_avg = maxdiff_cond['Oak17Acc'].mean() - mindiff_cond['Oak17Acc'].mean()
print('Estimated conditional expectation E[Attack|NumParams=min->max] {}'.format(cond_avg))

# mindiff_cond = df.loc[(np.abs(df['NumParams'] - min_numparams) < 0.0001)]
# maxdiff_cond = df.loc[(np.abs(df['NumParams'] - max_numparams) < 0.0001)]
# maxdiff_cond = df.loc[(np.abs(df['AccDiff'] - max_accdiff) < 0.001)]

# df.loc[(np.abs(df['AccDiff'] - 0.2) < 0.001)]
# df.loc[(np.abs(df['AccDiff'] - 0.2) < 0.1)]
# df.loc[(np.abs(df['AccDiff'] - 0.1) < 0.1)]
# df.loc[(np.abs(df['AccDiff'] - 0.1) < 0.01)]
# around01 = df.loc[(np.abs(df['AccDiff'] - 0.1) < 0.01)]
# around01[['NumParams','AccDiff','Oak17Acc']]
# print(around01)
