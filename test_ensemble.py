import unittest
import ensemble
import pandas as pd
import numpy as np

class TestScoreCombiner(unittest.TestCase):
    def setUp(self) -> None:
        self.df1 = pd.DataFrame(
            {
                'key': [1,2,3],
                'val1': [1,2,3],
                'dontcare': [7,8,9]
            }
        )
        self.df1_score_col = 'val1'
        
        self.df2 = pd.DataFrame(
            {
                'key': [2,3,4],
                'val2': [20,30,40],
            }
        )
        self.df2_score_col = 'val2'
        
        self.key = ['key']

    def test_add_score_one_df(self):
        ensembler = ensemble.ScoreCombiner()
        ensembler.add_score(
            self.df1, key=self.key, score_col=self.df1_score_col
        )
        expected = pd.DataFrame(
            {
                'key': [1,2,3],
                'val1': [1,2,3]
            }
        )
        self.assertTrue(ensembler.get_scores().equals(expected))
        self.assertEqual(ensembler.get_score_col_names(), [self.df1_score_col])

    def test_add_score_two_dfs(self):
        ensembler = ensemble.ScoreCombiner()
        ensembler.add_score(
            self.df1, key=self.key, score_col=self.df1_score_col
        )
        ensembler.add_score(
            self.df2, key=self.key, score_col=self.df2_score_col
        )        
        expected = pd.DataFrame(
            {
                'key': [1,2,3,4],
                'val1': [1,2,3,np.nan],
                'val2': [np.nan,20,30,40],
            }
        )
        self.assertTrue(ensembler.get_scores().equals(expected))

class TestEnsemble(unittest.TestCase):
    def test_rank_avg_ensemble(self):
        df = pd.DataFrame(
            {
                'x1': [3,2,1],
                'x2': [3,1,2]
            }
        )
        expected = pd.DataFrame(
            {
                'x1': [3,2,1],
                'x2': [3,1,2],
                'rank_avg': [1.0, 2.5, 2.5]
            }
        )
        res = ensemble.Ensemble.rank_avg_ensemble(
            df, variables=['x1','x2'], ensemble_score='rank_avg',
            ascending=False
        )
        self.assertTrue(res.equals(expected))

    def test_rank_avg_ensemble_with_ties(self):
        df = pd.DataFrame(
            {
                'x1': [3,2,1],
                'x2': [3,1,1]
            }
        )
        expected = pd.DataFrame(
            {
                'x1': [3,2,1],
                'x2': [3,1,1],
                'rank_avg': [1.0, 2.0, 2.5]
            }
        )
        res = ensemble.Ensemble.rank_avg_ensemble(
            df, variables=['x1','x2'], ensemble_score='rank_avg',
            ascending=False
        )
        self.assertTrue(np.array_equal(res, expected))

    def test_rank_avg_ensemble_mixed_ascending(self):
        df = pd.DataFrame(
            {
                'x1': [3,2,1],
                'x2': [1,2,3]
            }
        )
        expected = pd.DataFrame(
            {
                'x1': [3,2,1],
                'x2': [1,2,3],
                'rank_avg': [1.0, 2.0, 3.0]
            }
        )
        res = ensemble.Ensemble.rank_avg_ensemble(
            df, variables=['x1','x2'], ensemble_score='rank_avg',
            ascending=[False, True]
        )
        self.assertTrue(res.equals(expected))

    def test_rank_avg_ensemble_add_ranks(self):
        df = pd.DataFrame(
            {
                'x1': [3,2,1],
                'x2': [3,1,2]
            }
        )
        expected = pd.DataFrame(
            {
                'x1': [3,2,1],
                'x2': [3,1,2],
                'ens': [1.0, 2.5, 2.5],
                'x1_rank': [1.0,2.0,3.0],
                'x2_rank': [1.0,3.0,2.0]
            }
        )
        res = ensemble.Ensemble.rank_avg_ensemble(
            df, variables=['x1','x2'], ensemble_score='ens',
            ascending=False,
            add_ranks=True
        )
        self.assertTrue(res.equals(expected))

    def test_thresholded_avg(self):
        df = pd.DataFrame(
            {
                'x1': [-1, 0, 1],
                'x2': [3, 2, 1]
            }
        )
        res = ensemble.Ensemble.thresholded_avg(
            df, ['x1', 'x2'], ensemble_score='ens'
        )
        expected = [1.5, 1, 1.5]
        self.assertTrue(expected == res['ens'].to_list())

    def test_thresholded_avg_retain_all_columns(self):
        df = pd.DataFrame(
            {
                'x1': [-1, 0, 1],
                'x2': [1, 2, 3],
                'irrelevant': [42, 4.2, 1]
            }
        )
        res = ensemble.Ensemble.thresholded_avg(
            df, ['x1', 'x2'], ensemble_score='thresholded_avg'
        )
        self.assertTrue(
            list(res.columns) == ['x1', 'x2', 'irrelevant', 'thresholded_avg']
        )

    def test_thresholded_avg_not_change_variables(self):
        df = pd.DataFrame(
            {
                'x1': [-1, 0, 1],
                'x2': [1, 2, 3],
                'irrelevant': [42, 4.2, 1]
            }
        )
        res = ensemble.Ensemble.thresholded_avg(
            df, ['x1', 'x2'], ensemble_score='thresholded_avg'
        )
        self.assertTrue(
            df['x1'].to_list() == [-1, 0, 1]
        )

    def test_thresholded_avg_constant_input(self):
        df = pd.DataFrame(
            {
                'x1': [0, 0, 0],
                'x2': [0, 0, 0]
            }
        )
        res = ensemble.Ensemble.thresholded_avg(
            df, ['x1', 'x2'], ensemble_score='thresholded_avg'
        )
        expected = [0, 0, 0]
        self.assertTrue(expected == res['thresholded_avg'].to_list())

    def test_avg_ensemble(self):
        df = pd.DataFrame(
            {
                'x1': [1, 1, 1],
                'x2': [1, np.nan, 0]
            }
        )
        expected = [1, 1, 0.5]
        res = ensemble.Ensemble.avg_ensemble(
            df, ['x1', 'x2'], ensemble_score='ens'
        )
        self.assertTrue(expected == res['ens'].to_list())

    def test_maxpool_ensemble(self):
        df = pd.DataFrame(
            {
                'x1': [1, 1, 1],
                'x2': [1, np.nan, 10]
            }
        )
        expected = [1, 1, 10]
        res = ensemble.Ensemble.maxpool_ensemble(
            df, ['x1', 'x2'], ensemble_score='ens'
        )
        self.assertTrue(expected == res['ens'].to_list())

    def test_threshold_pruned_avg_ensemble(self):
        df = pd.DataFrame(
            {
                'x1': [1, 1, 1, 1],
                'x2': [1, np.nan, 0, 0.8]
            }
        )
        expected = [1, 1, 1, 0.9]
        res = ensemble.Ensemble.threshold_pruned_avg_ensemble(
            df, ['x1', 'x2'], ensemble_score='ens', pavg_thresh=0.7
        )
        self.assertTrue(expected == res['ens'].to_list())

    def test_top_k_pruned_avg_ensemble(self):
        df = pd.DataFrame(
            {
                'x1': [1, 1, 1, 1],
                'x2': [1, np.nan, 0, np.nan],
                'x3': [1, 2, 3, np.nan]
            }
        )
        expected = [1, 1.5, 2, 1]
        res = ensemble.Ensemble.top_k_pruned_avg_ensemble(
            df, ['x1', 'x2', 'x3'], ensemble_score='ens', pavg_k=2
        )
        self.assertTrue(expected == res['ens'].to_list())

class TestNormalization(unittest.TestCase):
    def test_minmax_norm(self):
        x = [1,2,3]
        expected = [0, 0.5, 1]
        res = ensemble.Normalize.minmax_norm(x)
        self.assertTrue(np.array_equal(res, expected))

    def test_minmax_norm_constant_input(self):
        x = [42, 42, 42]
        expected = [0, 0, 0]
        res = ensemble.Normalize.minmax_norm(x)
        self.assertTrue(np.array_equal(res, expected))

    def test_thresholded_std_norm(self):
        x = [-1, 0, 1]
        expected = [0, 0, 1/np.std(x)]
        res = ensemble.Normalize.thresholded_std_norm(x)
        self.assertTrue(np.array_equal(res, expected))
        
    def test_thresholded_std_norm_constant_input(self):
        x = [0, 0, 0]
        expected = [0, 0, 0]
        res = ensemble.Normalize.thresholded_std_norm(x)
        self.assertTrue(np.array_equal(res, expected))

    def test_thresholded_norm(self):
        x = [1, 2, 3, 4]
        expected = [2.5, 2.5, 3, 4]
        res = ensemble.Normalize.thresholded_norm(x)
        self.assertTrue(np.array_equal(res, expected))


if __name__ == '__main__':
    unittest.main()