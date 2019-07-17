import pandas as pd
import numpy as np

# To decide on whether to take Aria to the vet or not, you have the following
# information available to you.
# • Aria often howls if he is genuinely sick. However, Aria is a healthy dog and is
# only sick 5% of the time. If Aria is really sick, then he probably has not eaten
# much of his dinner and has food left in his bowl. In the past, when Aria is sick
# then he does not eat his dinner 60% of the time. However, when Aria is healthy
# he still does not eat his dinner 10% of the time.
# • Aria often howls if there is a full moon or your neighbour’s dog howls. Your
# neighbour’s dog sometimes howls at the full moon and sometimes howls when
# your neighbour is away. However, your neighbour’s dog is never affected by Aria’s
# howls. Since you live on Earth, there is a full moon once out of every twenty-eight
# days. You have observed that your neighbour travels three days out of every ten.
# • If there is no full moon and your neighbour is at home then your neighbour’s
# dog never howls. If there is no full moon and your neighbour is away then your
# neighbour’s dog howls half of the time. If there is a full moon and your neighbour
# is at home then your neighbour’s dog howls 40% of the time. Finally, if there is
# a full moon and your neighbour is away then your neighbour’s dog howls 80% of
# the time.
# • If all three triggers are there (Aria sick, full moon, and your neighbour’s dog
# howling), Aria howls 99% of the time. If none of the three triggers are there,
# Aria will not howl.
# • If Aria is sick he is more likely to howl. If being sick is the only trigger present,
# Aria will howl half of the time. If Aria is sick and your neighbour’s dog is howling
# then Aria will howl 75% of the time. If Aria is sick and there is a full moon, then
# Aria will howl 90% of the time.
# 4
# • If Aria is not sick then he is less likely to howl. The full moon and your neighbour’s
# dog will cause Aria to howl 65% of the time. If there is only a full moon then
# Aria will howl 40% of the time. If there is no full moon, but your neighbour’s
# dog is howling, then Aria will howl 20% of the time.

# • F H: Aria howls.
# • S: Aria is sick.
# • B: There is food left in Aria’s food bowl.
# • M: There is a full moon.
# • NA: Your neighbour’s away.
# • NH: Your neighbour’s dog howls.

# Question 1) Aria is howling. You look out the window and see that the moon is
# full. What is probability that Aria is sick?
#
# Question 2) (Aria is howling and the moon is still full.) You walk to the kitchen and
# see that Aria has not eaten and his food bowl is full. Given this new information,
# what is the probability that Aria is sick?
#
# Question 3) (Aria is howling, the moon is full and Aria has no eaten his dinner.)
# You decide to call your neighbour to see if they are home or not. The phone rings
# and rings so you conclude that your neighbour is away. Given this information,
# what is the probability that Aria is sick?

p_col_name = 'Prob'

factors=[   [[  'AS',   'Prob'],
             [  1,      0.05],
             [  0,      0.95]],

            [[  'M',    'Prob'],
             [  1,      0.035714286],
             [  0,      0.964285714]],

            [[  'NA',   'Prob'],
             [  1,      0.3],
             [  0,      0.7]],

            [[  'AB',   'AS',   'Prob'],
             [  1,      0,      0.1],
             [  1,      1,      0.6],
             [  0,      0,      0.9],
             [  0,      1,      0.4]],

            [[  'NH',   'M',    'NA',   'Prob'],
             [  1,      0,      0,      0.0],
             [  1,      0,      1,      0.5],
             [  1,      1,      0,      0.4],
             [  1,      1,      1,      0.8],
             [  0,      0,      0,      1.0],
             [  0,      0,      1,      0.5],
             [  0,      1,      0,      0.6],
             [  0,      1,      1,      0.2]],

            [[  'AH',   'AS',   'M',    'NH',   'Prob'],
             [  1,      1,      1,      1,      0.99],
             [  1,      0,      0,      0,      0.0],
             [  1,      1,      0,      0,      0.5],
             [  1,      1,      0,      1,      0.75],
             [  1,      1,      1,      0,      0.9],
             [  1,      0,      1,      1,      0.65],
             [  1,      0,      1,      0,      0.4],
             [  1,      0,      0,      1,      0.2],
             [  0,      1,      1,      1,      0.01],
             [  0,      0,      0,      0,      1.0],
             [  0,      1,      0,      0,      0.5],
             [  0,      1,      0,      1,      0.25],
             [  0,      1,      1,      0,      0.1],
             [  0,      0,      1,      1,      0.35],
             [  0,      0,      1,      0,      0.6],
             [  0,      0,      0,      1,      0.8]]
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

ordered_hidden_var_list = ['AB', 'AH', 'AS', 'M', 'NA', 'NH']


