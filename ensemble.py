import numpy as np
import pandas as pd
from typing import Union

class Normalize:
    def __init__(self) -> None:
        pass

    @staticmethod
    def minmax_norm(x):
        if np.max(x) - np.min(x) > 0:
            return (x-np.min(x))/(np.max(x)-np.min(x))
        else:
            return x-np.min(x)

    @staticmethod
    def std_norm(x):
        if np.std(x) > 0:
            return (x-np.mean(x))/np.std(x)
        else:
            return x-np.mean(x)

    @staticmethod
    def thresholded_std_norm(x):
        return np.maximum(Normalize.std_norm(x), 0)

    @staticmethod
    def thresholded_norm(x):
        return np.maximum(x, np.mean(x))

class ScoreCombiner:
    def __init__(self):
        self.scores = pd.DataFrame()
        self.score_col_names = []
    
    def add_score(self, df: pd.DataFrame, key: list, score_col: str):
        if len(self.scores):
            self.scores = self.scores.merge(
                df[key+[score_col]], left_on=key, right_on=key, how='outer'
            )
        else:
            self.scores = df[key+[score_col]]
        self.score_col_names.append(score_col)

    def get_scores(self):
        return self.scores

    def get_score_col_names(self):
        return self.score_col_names

class Ensemble:
    def __init__(self) -> None:
        pass

    @staticmethod
    def avg_ensemble(
        df, variables, ensemble_score='avg'
    ):
        df[ensemble_score] = df[variables].apply(np.mean, axis=1)
        return df

    @staticmethod
    def thresholded_avg(
        df, variables, ensemble_score='thresholded_avg'
    ):
        d = df.copy()
        d[variables] = d[variables].apply(Normalize().thresholded_norm)
        res = Ensemble.avg_ensemble(
            d,
            variables=variables,
            ensemble_score=ensemble_score
        )
        df[ensemble_score] = res[ensemble_score]
        return df

    @staticmethod
    def maxpool_ensemble(
        df, variables, ensemble_score='maxpool'
    ):
        df[ensemble_score] = df[variables].apply(np.max, axis=1)
        return df

    @staticmethod
    def threshold_pruned_avg_ensemble(
        df, variables, pavg_thresh=None, ensemble_score = 'threshold_pruned_avg'
    ):
        if pavg_thresh is None:
            pavg_thresh = 0.7*df[variables].max().max()
        df[ensemble_score] = (
            df[variables]
            .apply(lambda x: np.mean(x[x>pavg_thresh]), axis=1)
            .fillna(0)
        )
        return df

    @staticmethod
    def top_k_pruned_avg_ensemble(
        df, variables, pavg_k=None, ensemble_score = 'top_k_pruned_avg'
    ):
        if pavg_k is None:
            pavg_k = int(len(variables)/2)
        df[ensemble_score] = (
            df[variables].apply(lambda x: np.mean(x.nlargest(pavg_k)), axis=1)
        )
        return df

    @staticmethod
    def rank_avg_ensemble(
        df: pd.DataFrame, variables: list,
        ensemble_score: str = 'rank_avg',
        ascending: Union[bool, list] = False,
        add_ranks: bool = False
    ) -> pd.DataFrame:
        
        df['__index__'] = np.arange(len(df)) # used as key
        key = ['__index__']

        d = df.copy() # don't corrupt original df
  
        ensembler = ScoreCombiner()
        for i, var in enumerate(variables):
            ascnd = ascending if type(ascending) == bool else ascending[i]
            d[var] = d[var].rank(method='min', ascending=ascnd)
            ensembler.add_score(d, key, var)
    
        scores = Ensemble.avg_ensemble(
            ensembler.get_scores(), variables=variables,
            ensemble_score=ensemble_score
        )

        df = df.merge(
            scores[key+[ensemble_score]], left_on=key, right_on=key, how='left'
        )

        if add_ranks:
            for var in variables:
                df = df.merge(
                    scores[key+[var]].rename(columns={var: var+'_rank'}),
                    left_on=key,
                    right_on=key,
                    how='left'
                )

        df = df.drop(columns=key)

        return df

