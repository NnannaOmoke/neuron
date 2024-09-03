use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::Rng;

use crate::{base_array::BaseDataset, utils::model_selection::{TrainTestSplitStrategy, TrainTestSplitStrategyData}};

use super::classification_tree::RawClassificationTree;

pub enum MaxFeatureMode {
    Log,
    Sqrt
}
pub struct RandomForestClassifier {
    num_estimators: usize,
    trees: Vec<RawClassificationTree>,
    strategy: TrainTestSplitStrategy,
    data: TrainTestSplitStrategyData<f64, u32>,
    max_feature_mode: MaxFeatureMode
}

impl RandomForestClassifier {
    pub fn new(num_estimators: usize) -> Self {
        RandomForestClassifier {
            num_estimators,
            trees: vec![RawClassificationTree::default(); num_estimators],
            strategy: TrainTestSplitStrategy::default(),
            data: TrainTestSplitStrategyData::default(),
            max_feature_mode: MaxFeatureMode::Log
        }
    }
    pub fn strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        RandomForestClassifier {
            strategy,
            ..self
        }
    }
    pub fn bootstrap(features: ArrayView2<f64>, labels: ArrayView1<u32>, max_features: usize) -> (Array2<f64>, Array1<u32>) {
        let mut b_indices = vec![0usize; features.nrows()];
        let mut b_features = vec![0usize; max_features];
        let mut rng = rand::thread_rng();
        b_indices.iter_mut().for_each(|index| *index = rng.gen_range(0..features.nrows())); 
        b_features.iter_mut().for_each(|feature| *feature = rng.gen_range(0..features.ncols()));

        let mut out_features = Array2::<f64>::default((b_indices.len(), b_features.len()));
        let mut out_labels = Array1::<u32>::default(b_indices.len());
        //ri: row index, fi: feature index
        for ri in 0..features.nrows() {
            for fi in 0..max_features {
                out_features[(ri, fi)] = features[(b_indices[ri], b_features[fi])];
            }
        }
        for fi in 0..max_features {
            out_labels[fi] = labels[b_features[fi]];
        }
        (out_features, out_labels)
    }
    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        let target = dataset._get_string_index(target);
        self.data = TrainTestSplitStrategyData::<f64, u32>::new_c(dataset, target, self.strategy);
        self.internal_fit();
        
    }
    fn internal_fit(&mut self) {
        let (features, labels) = self.data.get_train();
        let max_features = match self.max_feature_mode {
            MaxFeatureMode::Log => usize::max(f64::log10(features.ncols() as f64) as usize, 1),
            MaxFeatureMode::Sqrt => usize::max(f64::sqrt(features.ncols() as f64) as usize, 1),
        };
        for tree in &mut self.trees {
            let (bootstrap_features, bootstrap_labels) = Self::bootstrap(features, labels, max_features);
            tree.fit(bootstrap_features.view(), bootstrap_labels.view());
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
    fn vote(results: ArrayView1<u32>) -> u32 {
        let mut highest_occurence = (0u32, 0usize); //(value, occurences)
        Self::for_each_unique(results.as_slice().unwrap(), |val| {
            let count = results.iter().filter(|&n| *n == val).count();
            if count > highest_occurence.1 {
                highest_occurence = (val, count);
            }
        });
        highest_occurence.0
    }
    pub fn predict(&self, data: ArrayView2<f64>) -> Array1<u32> {
        let mut results = Array2::<u32>::default((data.nrows(), 0));
        for tree in &self.trees {
            results.push_row(tree.predict(data).view());
        }
        let mut end_result = Array1::<u32>::default(data.nrows());
        let mut idx = 0;
        for result in results.rows() {
            end_result[idx] = Self::vote(result);
            idx += 1;
        }
        end_result
    }
    pub fn evaluate<F>(&self, function: F) -> Vec<f64>
    where
        F: Fn(ArrayView1<u32>, ArrayView1<u32>) -> Vec<f64>,
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

