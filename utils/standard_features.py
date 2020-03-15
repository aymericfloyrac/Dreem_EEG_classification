import numpy as np

def extract_std_ft(signal):
    """
    extracts sd, max, min, median of abs()
    """
    sd = np.std(signal)
    maxi = signal.max()
    mini = signal.min()
    med = np.median(np.abs(signal))
    return sd, maxi, mini, med

def extract_corr(signal_list):
    """
    extracts correlations between the first channel and the others
    """
    ref_signal = signal_list[0]
    correlations = []
    for i in range(1,len(signal_list)):
        correlations.append(np.corrcoef(ref_signal,signal_list[i])[0][1])

    return correlations

def extract_sleeve_corr(signal_list):
    """
    extracts correlations between the sleeves (sig**2)
    """
    sleeve_list = signal_list ** 2
    return extract_corr(sleeve_list)
