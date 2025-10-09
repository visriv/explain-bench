import matplotlib.pyplot as plt
import numpy as np

def plot_attributions(time, signal, attribution, title="Attribution"):
    plt.figure()
    plt.plot(time, signal)
    # overlay average attribution across features
    att = attribution.mean(axis=-1)
    plt.twinx()
    plt.plot(time, att)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Signal / Attribution")
    plt.tight_layout()
    return plt.gcf()
