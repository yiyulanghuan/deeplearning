# %%
# During eval Dropout is deactivated and just passes its input.
# During the training the probability p is used to drop activations. Also, the activations are scaled with 1./p as otherwise the expected values would differ between training and eval.

# https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615

import torch
import torch.nn as nn

drop = nn.Dropout()
x = torch.ones(1, 10)

# Train mode (default after construction)
drop.train()
print(drop(x))

# Eval mode
drop.eval()
print(drop(x))
