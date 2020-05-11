from sklearn.metrics import roc_curve,precision_recall_curve,auc
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Metrics(object):
    def __init__(self,scores,labels,name=None):
        self.scores=scores
        self.labels=labels
        self.__name__=name
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
    
    def compute_eer_point(self):
        eer_index=np.nanargmin(np.absolute((self.frr - self.fpr)))
        eer_threshold = self.thresholds[eer_index]
        frr_eer= self.frr[eer_index]
        tpr_eer=self.tpr[eer_index]
        return frr_eer,tpr_eer

    def compute_min_sum_point(self):
        min_sum_index=np.nanargmin(np.absolute((self.frr + self.fpr)))
        min_sum_threshold = self.thresholds[min_sum_index]
        min_sum_fpr=self.fpr[min_sum_index]
        min_sum_tpr=self.tpr[min_sum_index]
        return min_sum_fpr,min_sum_tpr


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
    def _set_title(self,base_title):
        try:
            self.is_defined("__name__")
            return " ".join([base_title,f"({self.__name__})"])
        except:
            return base_title

    def plot_metric(self,ax,metric,metrics_dict,opt=True):
        try:
            self.is_defined("metrics")
        except:
            self.log_metrics(metrics_dict)
        ax.plot(self.thresholds,self.metrics[metric])
        ax.set_xlabel("score")
        ax.set_ylabel("{} ".format(metric))
        title=self._set_title("{} as function of threshold".format(metric))
        ax.set_title(title)
        # plt.show()

        if opt:
            opt_id=self._get_opt_threshold(metric)

            print(f"Optimal {metric} value = {self.metrics[metric][opt_id]:.2f} given at threshold {self.thresholds[opt_id]:.2f}")

    def plot_far_frr(self,ax):
        try:
            self.is_defined("roc")
        except:
            self.get_roc_metrics()

        ax.plot(self.thresholds,self.frr,label="FRR")
        ax.plot(self.thresholds,self.tpr,label="FAR")
        ax.legend()
        ax.xlim(0,1)
        title=self._set_title("False Acceptance Rate (FAR) vs False Rejection Rate (FRR)")
        ax.set_title(title)

    def _add_eer_point(self,ax):
        fpr_eer,tpr_eer=self.compute_eer_point()
        ax.plot(fpr_eer,tpr_eer,"or")
        ax.annotate('EER ',
            xy=(fpr_eer, tpr_eer), xycoords='data',
            xytext=(5, 5), textcoords='offset points',
            horizontalalignment='right', verticalalignment='bottom')

    def _add_min_sum_point(self,ax):
        min_sum_fpr,min_sum_tpr=self.compute_min_sum_point()
        ax.plot(min_sum_fpr,min_sum_tpr,"or")
        ax.annotate('Minimal sum ',
            xy=(min_sum_fpr, min_sum_tpr), xycoords='data',
            xytext=(70, -35), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.1),
            horizontalalignment='right', verticalalignment='bottom')

    def plot_roc_curve(self,ax,point=""):
        """plot the ROC curve
             (TPR against the FPR for different threshold values)
             point=="EER" if add EER point to plot
             point == "min_sum" if add far,frr min sum point to plot"""
        try:
            self.is_defined("roc")
        except:
            self.get_roc_metrics()
        
        roc_auc=auc(self.fpr,self.tpr)
        ax.plot(self.fpr, self.tpr, color='darkorange',
                lw=2, label='ROC curve (auc = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        title=self._set_title('Fingerprint detector ROC curve')
        ax.set_title(title)
        if (point=="EER"):
            self._add_eer_point(ax)
        if(point=="min_sum"):
            self._add_min_sum_point(ax)
        ax.legend(loc="lower right")
        # plt.show()

    def plot_det_curve(self,ax):
        """plot the DET curve 
        (FRR (=1-tpr) against the FAR for different threshold values)"""
        try:
            self.is_defined("roc")
        except:
            self.get_roc_metrics()
        ax.plot(self.frr, self.tpr, color='darkorange',
                lw=2, label='DET curve ')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Rejection Rate')
        ax.set_ylabel('False Acceptance Rate')
        title=self._set_title('Fingerprint detector DET curve')
        ax.set_title(title)
        ax.legend(loc="upper right")
        # plt.show()

    def plot_pr_curve(self,ax):
        """Calculate and plot the Precision-Recall curve for this system"""
        try:
            self.is_defined("pr")
        except:
            self.get_pr_metrics()

        pr_auc=auc(self.recall,self.precision)
        ax.plot(self.precision, self.recall, color='darkorange',
                lw=2, label='Precision-Recall curve (auc = %0.2f)' % pr_auc )
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        title=self._set_title('Fingerprint detector Precision-Recall curve')
        ax.set_title(title)
        ax.legend(loc="lower right")
        # plt.show()