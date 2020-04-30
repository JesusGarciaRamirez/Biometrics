from sklearn.metrics import roc_curve,precision_recall_curve,auc
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Metrics(object):
    def __init__(self,scores,labels):
        self.scores=scores
        self.labels=labels
        self.thresholds=None
        self.tpr=None
        self.fpr=None
        self.frr=None

    def _get_metric(self,metric):
        return metric(self.labels,self.scores)

    def is_defined(self,argument):
        if(getattr(self,argument)==None):
            raise ValueError

    def get_roc_metrics(self):
        self.fpr, self.tpr, self.thresholds = self._get_metric(metric=roc_curve)
        self.frr=1-self.tpr
        #Flag to specify that roc related metrics have been already computed
        self.roc=True

    def get_pr_metrics(self):
        self.precision, self.recall, self.thresholds = precision_recall_curve(self.labels,self.scores)
        #Flag to specify that Precision and recall have been already computed
        self.pr=True

    def _init_metrics(self,metrics_dict):
        self.metrics={}
        for key in metrics_dict.keys():
            self.metrics[key]=[]

    def _scores_to_preds(self,threshold):
        y_pred=np.empty(len(self.scores))
        cond=(self.scores < threshold)
        y_pred[cond]=0
        y_pred[~cond]=1
        return y_pred
    
    def _get_opt_threshold(self,metric):
        try:
            self.is_defined("metrics")
        except:
           raise ValueError ("You haven´t computed the metrics dict")

        return next(idx for idx in range(len(self.thresholds))
                        if self.metrics[metric][idx]==max(self.metrics[metric]))
     
              
    def log_metrics(self,metrics_dict):
        #Inits
        self._init_metrics(metrics_dict)
        try:
            self.is_defined("thresholds")
        except:
            self.thresholds=np.linspace(0,1,50)

        for threshold in tqdm(self.thresholds):
            #create y_pred
            y_preds=self._scores_to_preds(threshold)
            #metrics
            for metric_name,metric in metrics_dict.items():
                metric_score=metric(self.labels,y_preds)
                self.metrics[metric_name].append(metric_score)

        return self.metrics

    def plot_metric(self,metric,metrics_dict,opt=True):
        try:
            self.is_defined("metrics")
        except:
            self.log_metrics(metrics_dict)
        plt.plot(self.thresholds,self.metrics[metric])
        plt.xlabel("score")
        plt.ylabel("{} ".format(metric))
        plt.title("{} as function of threshold".format(metric))
        plt.show()

        if opt:
            opt_id=self._get_opt_threshold(metric)

            print("Optimal {} value = {} given at threshold {}".format(metric,
                                                                self.metrics[metric][opt_id],
                                                                self.thresholds[opt_id]))
    def plot_far_frr(self):
        try:
            self.is_defined("roc")
        except:
            self.get_roc_metrics()

        plt.plot(self.thresholds,self.frr,label="FRR")
        plt.plot(self.thresholds,self.tpr,label="FAR")
        plt.legend()
        plt.xlim(0,1)
        plt.title("False Acceptance Rate (FAR) vs False Rejection Rate (FRR)")
        plt.show()

    def plot_roc_curve(self):
        """plot the ROC curve
             (TPR against the FPR for different threshold values)"""
        try:
            self.is_defined("roc")
        except:
            self.get_roc_metrics()
        
        roc_auc=auc(self.fpr,self.tpr)
        plt.plot(self.fpr, self.tpr, color='darkorange',
                lw=2, label='ROC curve (auc = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Fingerprint detector ROC curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_det_curve(self):
        """plot the DET curve 
        (FRR (=1-tpr) against the FAR for different threshold values)"""
        try:
            self.is_defined("roc")
        except:
            self.get_roc_metrics()
        plt.plot(self.frr, self.tpr, color='darkorange',
                lw=2, label='DET curve ')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Rejection Rate')
        plt.ylabel('False Acceptance Rate')
        plt.title('Fingerprint detector DET curve')
        plt.legend(loc="upper right")
        plt.show()

    def plot_pr_curve(self):
        """Calculate and plot the Precision-Recall curve for this system"""
        try:
            self.is_defined("pr")
        except:
            self.get_pr_metrics()

        pr_auc=auc(self.recall,self.precision)
        plt.plot(self.precision, self.recall, color='darkorange',
                lw=2, label='Precision-Recall curve (auc = %0.2f)' % pr_auc )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Fingerprint detector Precision-Recall curve')
        plt.legend(loc="lower right")
        plt.show()
