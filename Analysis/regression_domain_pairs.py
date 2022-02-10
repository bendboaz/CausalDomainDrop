import glob

import pandas as pd
import statsmodels.api as sm

from paths import EXPERIMENTS_ROOT as experiments_dir

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

all_pairs = glob.glob(experiments_dir + '*.csv')

x_cols = ['source', 'target', 'task', 'tc', 'cc', 'modified_source_f1',
          'modified_source_acc', 'source_f1', 'source_acc', 'target_f1',
          'target_acc']
reg_cols = ['ate_performance_1', 'ate_performance_2', 'ate_performance_3', 'performance_degradation']

df = pd.DataFrame(columns=reg_cols)
for pair in all_pairs:
    ate_performances = []
    cur_df = pd.read_csv(pair)
    tcs = cur_df['tc'].unique()
    for tc in tcs:
        ate_performance = cur_df[(cur_df['tc'] == tc) & (cur_df['task'] == 'MLM')]['modified_source_f1'].values[0] - \
                          cur_df[(cur_df['tc'] == tc) & (cur_df['task'] == 'CAUSALM')]['modified_source_f1'].values[0]
        ate_performances.append(ate_performance)
    performance_degradation = (cur_df['source_f1'] - cur_df['target_f1']).values[0]

    if len(ate_performances) > 1:
        df = df.append(pd.DataFrame([ate_performances + [performance_degradation]], columns=reg_cols),
                       ignore_index=True)

print(df)
print(df.corr())

# Add intercept
X = sm.add_constant(df[['ate_performance_1', 'ate_performance_2', 'ate_performance_3']], prepend=False)
X = df[['ate_performance_1', 'ate_performance_2', 'ate_performance_3']]
y = df['performance_degradation']

model = sm.OLS(y, X)
reg_results = model.fit()
print(reg_results.summary())
