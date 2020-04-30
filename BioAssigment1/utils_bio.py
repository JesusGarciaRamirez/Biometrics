import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma, norm
from sklearn.metrics import accuracy_score, f1_score


def plot_dist(data,scaler,distribution,title=""):
    xmin,xmax=min(data),max(data)
    # Fit a distribution to the data:
    params=distribution.fit(data)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    x = np.linspace(xmin, xmax, 100)

    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)

    # # Plot the histogram.
    counts, bins = np.histogram(data)
    total=sum(counts)
    counts=counts/(total * (bins[1]-bins[0]))
    bins=scaler.transform(bins.reshape(-1,1)).flatten()
    plt.hist(bins[:-1], bins, weights=counts)
    x=scaler.transform(x.reshape(-1,1)).flatten()
    plt.plot(x, pdf, 'k', linewidth=2)

    plt.title(title)

    plt.show()
    return pdf,x

def plot_joint_dist(genuine_dict,impostor_dict,x_lim=(0,0.11)):
    #joint distributions plot
    plt.fill(genuine_dict["points"],genuine_dict["dist"], facecolor="blue",alpha=0.5,label="Genuine")
    plt.fill(impostor_dict["points"],impostor_dict["dist"] ,facecolor="red",alpha=0.5,label="Impostor")
    plt.xlim(x_lim)
    plt.title("Impostor and Genuine Distributions")
    plt.legend()
    plt.show()


