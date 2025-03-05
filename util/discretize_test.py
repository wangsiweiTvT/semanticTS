from BPEutil import discretize_series
from BPEutil import restore_series
import numpy as np


ts = [1,2,3,4,5,6]
num_bins= 6
symbols = discretize_series(ts,num_bins)
print(symbols)
bins = np.linspace(min(ts), max(ts), num_bins +1)
print(bins)

recoverts = restore_series(symbols,bins)
print(recoverts)

