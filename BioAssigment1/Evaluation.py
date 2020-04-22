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

def scores_to_preds(scores,threshold):
    y_pred=np.empty(len(scores))
    cond=(scores < threshold)
    y_pred[cond]=0
    y_pred[~cond]=1
    return y_pred

def init_metrics(metrics_dict):
    metrics={}
    for key in metrics_dict.keys():
        metrics[key]=[]
    return metrics

def log_metrics(scores,thresholds,metrics_dict):
    metrics=init_metrics(metrics_dict)
    for threshold in thresholds:
        #create y_pred
        y_preds=scores_to_preds(scores,threshold)
        #metrics
        for metric_name,metric in metrics_dict.items():
            metric_score=metric(labels,y_preds)
            metrics[metric_name].append(metric_score)
    return metrics

def plot_metric(scores,metrics,metric):
    plt.plot(thresholds,metrics[metric])
    plt.xlabel("score")
    plt.ylabel("{} ".format(metric))
    plt.title("{} as function of threshold".format(metric))
    plt.show()
    
def get_opt_threshold(tresholds,metrics,metric):
    return next(idx for idx in range(len(thresholds))
                      if metrics[metric][idx]==max(metrics[metric]))



#get f1 and accuracy as a funtion of thresholds
metrics=log_metrics(scores,
                    thresholds,
                    metrics_dict   
                )
def get_metric(labels,scores,metric):
    return metric(labels,scores)