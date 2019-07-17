import pandas as pd
import numpy as np

# Mr. Holmes lives in a high crime area and therefore has installed a burglar alarm. He relies on
# his neighbors to phone him when they hear the alarm sound. Mr. Holmes has two neighbors, Dr.
# Watson and Mrs. Gibbon.
# Unfortunately, his neighbors are not entirely reliable. Dr. Watson is known to be a tasteless
# practical joker and Mrs. Gibbon, while more reliable in general, has occasional drinking problems.
# Mr. Holmes also knows from reading the instruction manual of his alarm system that the device is
# sensitive to earthquakes and can be triggered by one accidentally. He realizes that if an earthquake
# has occurred, it would surely be on the radio news.

# Question 1) What is the probability that a burglary is happening given that Dr. Watson and Mrs. Gibbon
# both call? In other words, what is P(BjW ^ G)?

p_col_name = 'Prob'

factors=[   [[  'E',    'Prob'],
             [  1,      0.0003],
             [  0,      0.9997]],

            [[  'B',    'Prob'],
             [  1,      0.0001],
             [  0,      0.9999]],

            [[  'A',    'B',    'E',     'Prob'],
             [  1,      1,      1,      0.96],
             [  0,      1,      1,      0.04],
             [  1,      1,      0,      0.95],
             [  0,      1,      0,      0.05],
             [  1,      0,      1,      0.2],
             [  0,      0,      1,      0.8],
             [  1,      0,      0,      0.01],
             [  0,      0,      0,      0.99]],

            [[ 'W',    'A',    'Prob'],
             [  1,      1,      0.8],
             [  0,      1,      0.2],
             [  1,      0,      0.4],
             [  0,      0,      0.6]],

            [[ 'G',    'A',    'Prob'],
             [  1,      1,      0.4],
             [  0,      1,      0.6],
             [  1,      0,      0.04],
             [  0,      0,      0.96]],
        ]

# Make Pandas dataframe from above python list
for index, factor in enumerate(factors):
    factor = np.array(factor)
    df = pd.DataFrame(  data=factor[1:,:],
                        columns=factor[0,:])

    # set correct types for each column
    df[p_col_name] = df[p_col_name].astype(float)
    columns = list(df)
    columns.remove(p_col_name)
    for col in columns:
        df[col] = df[col].astype(int)

    factors[index] = df

ordered_hidden_var_list = ['W', 'G', 'A', 'B', 'E']


