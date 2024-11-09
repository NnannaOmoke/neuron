use core::num;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::Rng;

use crate::{
    base_array::BaseDataset,
    utils::model_selection::{TrainTestSplitStrategy, TrainTestSplitStrategyData},
};

use super::regression_tree::RawRegressionTree;

pub enum MaxFeatureMode {
    Log,
    Sqrt,
}
pub struct RandomForestRegressor {
    num_estimators: usize,
    trees: Vec<(RawRegressionTree, Vec<usize>)>, //Vec<usize> for feature indices
    strategy: TrainTestSplitStrategy,
    data: TrainTestSplitStrategyData<f64, f64>,
    max_feature_mode: MaxFeatureMode,
}

impl RandomForestRegressor {
    pub fn new(num_estimators: usize) -> Self {
        RandomForestRegressor {
            num_estimators,
            trees: vec![(RawRegressionTree::new(), vec![]); num_estimators],
            strategy: TrainTestSplitStrategy::default(),
            data: TrainTestSplitStrategyData::default(),
            max_feature_mode: MaxFeatureMode::Sqrt,
        }
    }
    pub fn strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        RandomForestRegressor { strategy, ..self }
    }
    pub fn bootstrap(
        features: ArrayView2<f64>,
        labels: ArrayView1<f64>,
        max_features: usize,
    ) -> (Array2<f64>, Array1<f64>, Vec<usize>) {
        let mut b_indices = vec![0usize; features.nrows()];
        let mut b_features = vec![0usize; max_features];
        let mut rng = rand::thread_rng();
        b_indices
            .iter_mut()
            .for_each(|index| *index = rng.gen_range(0..features.nrows()));
        b_features
            .iter_mut()
            .for_each(|feature| *feature = rng.gen_range(0..features.ncols()));

        let mut out_features = Array2::<f64>::default((b_indices.len(), b_features.len()));
        let mut out_labels = Array1::<f64>::default(b_indices.len());
        //ri: row index, fi: feature index
        for ri in 0..features.nrows() {
            for fi in 0..max_features {
                out_features[(ri, fi)] = features[(b_indices[ri], b_features[fi])];
            }
            out_labels[ri] = labels[b_indices[ri]];
        }
        (out_features, out_labels, b_features)
    }
    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        let target = dataset._get_string_index(target);
        self.data = TrainTestSplitStrategyData::<f64, f64>::new_r(dataset, target, self.strategy);
        self.internal_fit();
    }
    fn internal_fit(&mut self) {
        let (features, labels) = self.data.get_train();
        let max_features = match self.max_feature_mode {
            MaxFeatureMode::Log => usize::max(f64::log10(features.ncols() as f64) as usize, 1),
            MaxFeatureMode::Sqrt => usize::max(f64::sqrt(features.ncols() as f64) as usize, 1),
        };
        for tree in &mut self.trees {
            let (bootstrap_features, bootstrap_labels, bootstrap_feature_indices) =
                Self::bootstrap(features, labels, max_features);
            tree.0.fit(bootstrap_features.view(), bootstrap_labels.view());
            tree.1 = bootstrap_feature_indices;
        }
    }
    fn for_each_unique<T>(array: &[T], mut func: impl FnMut(T) -> ())
    where
        T: PartialEq + Copy,
    {
        let mut encountered: Vec<T> = vec![];
        for val in array {
            if !encountered.contains(val) {
                func(*val);
                encountered.push(*val);
            }
        }
    }
    fn vote(results: ArrayView1<f64>) -> f64 {
        let mut sum = 0.0;
        for elem in &results {
            sum += elem;
        }
        sum / results.len() as f64
    }

    pub fn predict(&self, data: ArrayView2<f64>) -> Array1<f64> {
        let mut results = Array2::default((data.nrows(), self.trees.len()));
        self.trees.iter().enumerate().for_each(|(index, tree)| {
            //is there a way to just view these instead of copying
            let mut data_for_tree = Array2::default((data.nrows(), tree.1.len()));
            for (idx, col) in tree.1.iter().enumerate() {
                data_for_tree.column_mut(idx).assign(&data.column(*col));
            }
            results.column_mut(index).assign(&tree.0.predict(data_for_tree.view()).view());
        });
        let mut result = Array1::default(data.nrows());
        results
            .rows()
            .into_iter()
            .enumerate()
            .for_each(|(index, res_row)| {
                result[index] = Self::vote(res_row);
            });
            
        result
    }

    pub fn evaluate<F>(&self, function: F) -> Vec<f64>
    where
        F: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Vec<f64>,
    {
        let (features, ground_truth) = match self.strategy {
            TrainTestSplitStrategy::None => {
                //get train data
                self.data.get_train()
            }
            TrainTestSplitStrategy::TrainTest(_) => self.data.get_test(),
            TrainTestSplitStrategy::TrainTestEval(_, _, _) => self.data.get_eval(),
        };
        let preds = self.predict(features);
        function(ground_truth, preds.view())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_array::BaseDataset;
    use crate::utils::metrics::root_mean_square_error;
    use std::path::Path;
    #[test]
    fn test_random_forest_r() {
        let dataset = BaseDataset::from_csv(
            Path::new("src/base_array/test_data/boston.csv"),
            true,
            true,
            b',',
        )
        .unwrap();
        let mut tree =
            RandomForestRegressor::new(30).strategy(TrainTestSplitStrategy::TrainTest(0.7));
        tree.fit(&dataset, "MEDV");
        
        let rmse = tree.evaluate(root_mean_square_error);
        dbg!(rmse);
    }
}