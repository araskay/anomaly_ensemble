import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
import seaborn as sns
import sklearn.metrics as skm

class Plotter:
    @staticmethod
    def cluster_mat(corrmat, fig_size=(6,4)):
        corrmat_abs = np.abs(corrmat)
        d = sch.distance.pdist(corrmat_abs)
        z = sch.linkage(d, method='complete')
        plt.figure(figsize=fig_size)
        dend = sch.dendrogram(
            z, labels=corrmat_abs.columns, orientation='top', leaf_rotation=90
        )
        plt.show()
        
        corrmat_reord_cols = corrmat[dend['ivl']]
        corrmat_reord_cols_reord_rows = corrmat_reord_cols.reindex(dend['ivl'])

        plt.figure(figsize=fig_size)
        sns.heatmap(corrmat_reord_cols_reord_rows, center=0, vmin=-1, vmax=1)
        plt.xticks(rotation=45, ha='right')
        plt.show()
        
        return corrmat_reord_cols_reord_rows
    
    @staticmethod
    def clustered_hmap(df, variables, fig_size=(6,4)):
        # running a hierarchical clustering on the original correlation vlaues
        # can result in negative correlations to be misplaced.
        corrmat = df[variables].corr()
    
        corrmat_reord_cols_reord_rows = Plotter.cluster_mat(
            corrmat, fig_size=fig_size
        )

        return corrmat, corrmat_reord_cols_reord_rows

    @staticmethod
    def plot_roc(y_true, y_score, title=''):
        fpr, tpr, thresholds = skm.roc_curve(y_true=y_true, y_score=y_score)
        auc = skm.roc_auc_score(y_true=y_true, y_score=y_score)
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(title+' (AUC = {})'.format(auc))
        plt.show()
        return auc