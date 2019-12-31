# %%
import pandas as pd

import holoviews as hv
hv.extension('bokeh', 'matplotlib', width=100)
import hvplot.pandas

# %%
train_valid = pd.read_csv("./train_valid.csv")

# %%
train_valid.hvplot(x='Step', y=['Train Accuracy of Vanilla CNN', 'Train Accuracy of Modern CNN', 'Train Accuracy of SqueezeNet', 'Train Accuracy of Vanilla Complex CNN'], xlabel='Epoch Steps', ylabel='Accuracy Rate', height=600, width=800, legend='bottom_right')

# %%
train_valid.hvplot(x='Step', y=['Valid Accuracy of Vanilla CNN', 'Valid Accuracy of Modern CNN', 'Valid Accuracy of SqueezeNet', 'Valid Accuracy of Vanilla Complex CNN'], xlabel='Epoch Steps', ylabel='Accuracy Rate', height=600, width=800, legend='bottom_right')

