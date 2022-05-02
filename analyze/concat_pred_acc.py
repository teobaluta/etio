import sys
import pandas as pd

pred_acc_1 = 'results-cont-wd_0.000500.csv'
pred_acc_2 = 'results-cont-wd_0.005000.csv'


# Saving files for paper
formats = {'Max CorrHWD': '{:10.4f}', 'Max CorrNWD': '{:10.4f}',
            'PredCor-w_dkHWD': '{:10.4f}', 'PredCor-w_dkNWD': '{:10.4f}',
           'MSE-w_dkHWD': '{:.2E}', 'MSE-w_dkNWD': '{:.2E}'}

df1 = pd.read_csv(pred_acc_1)
df1 = df1.loc[:, ~df1.columns.str.contains('^Unnamed')]

df2 = pd.read_csv(pred_acc_2)
df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]

result = df1.join(df2.set_index(['Loss','Attack']), on=['Loss','Attack'], lsuffix='NWD', rsuffix='HWD')

mean_mse = pd.DataFrame()
mean_mse = pd.concat([mean_mse, result['MSE-w_dkHWD'], result['MSE-w_dkNWD']])
print('Mean MSE {}'.format(mean_mse.mean()))

mean_corr = pd.DataFrame()
mean_corr = pd.concat([mean_corr, result['Max CorrHWD'], result['Max CorrNWD']])
print('Mean Corr {}'.format(mean_corr.mean()))

thresh = 0.03
count_nwd = result.loc[result['PredCor-w_dkNWD'] - result['Max CorrNWD'] >= thresh].count()
count_hwd = result.loc[result['PredCor-w_dkHWD'] - result['Max CorrHWD'] >= thresh].count()
print("Better than {} = {}".format(thresh, count_nwd + count_hwd))

for col, f in formats.items():
    result[col] = result[col].map(lambda x: f.format(x))
result.to_csv('result-pred_cor.csv')
