use core::{f64, num};
use std::collections::HashMap;

use naga::proc::index;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::{
    base_array::BaseDataset,
    utils::{
        model_selection::{TrainTestSplitStrategy, TrainTestSplitStrategyData},
        scaler::Scaler,
        scaler::ScalerState,
    },
};

#[derive(Debug, Clone, Default)]
struct RegressionTreeNode {
    left: Option<Box<RegressionTreeNode>>,
    right: Option<Box<RegressionTreeNode>>,
    feature_idx: usize,
    gain: f64,
    value: f64,
    threshold: f64,
}

#[derive(Debug, Clone)]
pub struct RawRegressionTree {
    root: Option<Box<RegressionTreeNode>>,
    min_samples: usize,
    max_depth: usize,
}

impl RawRegressionTree {
    pub fn fit(&mut self, features: ArrayView2<f64>, labels: ArrayView1<f64>) {
        self.root =
            Some(self.build_internal(features, labels, Vec::from_iter(0..features.nrows()), 0));
    }
    pub fn new() -> Self {
        RawRegressionTree {
            root: None,
            min_samples: 10,
            max_depth: 10
        }
    }
    pub fn predict(&self, data: ArrayView2<f64>) -> Array1<f64> {
        let mut array = Array1::<f64>::default(data.nrows());
        let mut idx = 0usize;
        match &self.root {
            None => array,
            Some(node) => {
                for row in data.rows() {
                    array[idx] = node.predict(row);
                    idx += 1;
                }
                array
            }
        }
    }

    fn split_data(
        &self,
        dataset: ArrayView2<f64>,
        indices: &[usize],
        feature_idx: usize,
        threshold: f64,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left = Vec::<usize>::new();
        let mut right = Vec::<usize>::new();
        for row in indices {
            if *dataset.get((*row, feature_idx)).unwrap() <= threshold {
                left.push(*row);
            } else {
                right.push(*row);
            }
        }
        (left, right)
    }

    fn mse(data: &[f64]) -> f64 {
        let mut error = 0.0;
        let mut avg = 0.0;
        for elem in data {
            avg += elem;
        }
        avg /= data.len() as f64;
        for elem in data {
            error += (elem - avg) * (elem - avg);
        }
        error /= data.len() as f64;
        error
    }

    fn impurity_gain(parent: &[f64], left: &[f64], right: &[f64]) -> f64 {
        let parent_mse = Self::mse(parent);
        let mse_left = Self::mse(left);
        let mse_right = Self::mse(right);
        let w_left = left.len() as f64 / parent.len() as f64;
        let w_right = right.len() as f64 / parent.len() as f64;
        parent_mse - w_left * mse_left + w_right * mse_right
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

    fn column_from_index_list(
        dataset: ArrayView2<f64>,
        index_list: &[usize],
        col_index: usize,
    ) -> Vec<f64> {
        let column = dataset.column(col_index);
        Self::pick_values(column, index_list)
    }

    fn pick_values<T>(array: ArrayView1<T>, index_list: &[usize]) -> Vec<T>
    where
        T: Copy,
    {
        let mut out_column = Vec::<T>::new();
        for index in index_list {
            out_column.push(array[*index]);
        }
        out_column
    }

    fn best_split(
        &self,
        dataset: ArrayView2<f64>,
        target: ArrayView1<f64>,
        indices: &[usize],
    ) -> Split {
        let num_features = dataset.ncols();
        let mut best_split = Split::new();
        for feature_idx in 0..num_features {
            let feature = Self::column_from_index_list(dataset, &indices, feature_idx);
            Self::for_each_unique(&feature, |val| {
                let (left, right) = self.split_data(dataset, indices, feature_idx, val);
                if !left.is_empty() && !right.is_empty() {
                    let y = Self::pick_values(target, indices);
                    let left_y = Self::pick_values(target, &left);
                    let right_y = Self::pick_values(target, &right);

                    let info_gain = Self::impurity_gain(&y, &left_y, &right_y);
                    if info_gain > best_split.gain {
                        best_split.gain = info_gain;
                        best_split.feature_idx = feature_idx;
                        best_split.threshold = val;
                        best_split.left = left;
                        best_split.right = right;
                    }
                }
            });
        }
        best_split
    }

    fn leaf_value(y: Vec<f64>) -> f64 {
        let mut val = 0.0;
        for elem in &y {
            val += elem;
        }
        val / y.len() as f64
    }

    fn build_internal(
        &mut self,
        features: ArrayView2<f64>,
        labels: ArrayView1<f64>,
        index_list: Vec<usize>,
        current_depth: usize,
    ) -> Box<RegressionTreeNode> {
        let num_samples = features.nrows();
        if num_samples >= self.min_samples && current_depth <= self.max_depth {
            let best_split = self.best_split(features, labels, &index_list);
            if best_split.gain != 0.0 {
                let left =
                    self.build_internal(features, labels, best_split.left, current_depth + 1);
                let right =
                    self.build_internal(features, labels, best_split.right, current_depth + 1);
                return Box::new(RegressionTreeNode {
                    left: Some(left),
                    right: Some(right),
                    feature_idx: best_split.feature_idx,
                    gain: best_split.gain,
                    threshold: best_split.threshold,
                    value: f64::NAN,
                });
            }
        }
        let value = Self::leaf_value(Self::pick_values(labels, &index_list));
        Box::new(RegressionTreeNode {
            left: None,
            right: None,
            feature_idx: usize::MAX,
            gain: f64::NAN,
            threshold: f64::NAN,
            value,
        })
    }
}

pub struct RegressionTreeBuilder {
    target_col: usize,
    strategy: TrainTestSplitStrategy,
    data: TrainTestSplitStrategyData<f64, f64>,
    scaler: ScalerState,
    tree: RawRegressionTree,
}
struct Split {
    gain: f64,
    threshold: f64,
    feature_idx: usize,
    left: Vec<usize>,
    right: Vec<usize>,
}
impl Split {
    fn new() -> Self {
        Self {
            gain: 0.0,
            feature_idx: 0,
            threshold: 0.0,
            left: vec![],
            right: vec![],
        }
    }
}

impl RegressionTreeBuilder {
    pub fn new() -> Self {
        Self {
            target_col: 0,
            strategy: TrainTestSplitStrategy::None,
            data: TrainTestSplitStrategyData::default(),
            scaler: ScalerState::default(),
            tree: RawRegressionTree::new(),
        }
    }

    pub fn strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        Self { strategy, ..self }
    }

    pub fn scaler(self, scaler: ScalerState) -> Self {
        Self { scaler, ..self }
    }

    pub fn min_samples(self, min_samples: usize) -> Self {
        let prev = self.tree;
        Self {
            tree: RawRegressionTree {
                min_samples,
                ..prev
            },
            ..self
        }
    }

    pub fn max_depth(self, max_depth: usize) -> Self {
        let prev = self.tree;
        Self {
            tree: RawRegressionTree { max_depth, ..prev },
            ..self
        }
    }

    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target_col = dataset._get_string_index(target);
        self.data =
            TrainTestSplitStrategyData::<f64, f64>::new_r(dataset, self.target_col, self.strategy);
        let mut scaler = Scaler::from(&self.scaler);
        scaler.fit(self.data.get_train().0);
        scaler.transform(&mut self.data.get_train_mut().0);
        match self.strategy {
            TrainTestSplitStrategy::TrainTest(_) => {
                scaler.transform(&mut self.data.get_test_mut().0);
            }
            TrainTestSplitStrategy::TrainTestEval(_, _, _) => {
                scaler.transform(&mut self.data.get_test_mut().0);
                scaler.transform(&mut self.data.get_test_mut().0);
            }
            _ => {}
        };
        self.internal_fit();
    }

    pub fn internal_fit(&mut self) {
        let (features, labels) = self.data.get_train();
        self.tree.fit(features, labels);
    }

    pub fn predict(&self, data: ArrayView2<f64>) -> Array1<f64> {
        self.tree.predict(data)
    }

    pub fn evaluate<F>(&self, function: F) -> Vec<f64>
    //using a vec because user evaluation functions might return maybe one value or three
    //all the functions we plan to build in will only return one value, however
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

    pub fn raw_mut(&mut self) -> &mut RawRegressionTree {
        &mut self.tree
    }
}
impl RegressionTreeNode {
    fn predict(&self, data: ArrayView1<f64>) -> f64 {
        if self.left.is_some() && self.right.is_some() {
            if data[self.feature_idx] <= self.threshold {
                self.left.as_ref().unwrap().predict(data)
            } else {
                self.right.as_ref().unwrap().predict(data)
            }
        } else {
            self.value
        }
    }
    fn display(&self, depth: usize) {
        if self.left.is_some() && self.right.is_some() {
            println!(
                "{} feature: {} gain: {} threshold: {}",
                "  ".repeat(depth),
                self.feature_idx,
                self.gain,
                self.threshold
            );
            println!("{} left:", "  ".repeat(depth));
            self.left.as_ref().unwrap().display(depth + 1);
            println!("{} right:", "  ".repeat(depth));
            self.right.as_ref().unwrap().display(depth + 1);
        } else {
            println!("{} value: {}", "  ".repeat(depth), self.value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_array::BaseDataset;
    use crate::utils::metrics::root_mean_square_error;
    use crate::utils::model_selection::TrainTestSplitStrategy;
    use std::path::Path;
    #[test]
    fn test_tree_r() {
        let dataset = BaseDataset::from_csv(
            Path::new("src/base_array/test_data/boston.csv"),
            true,
            true,
            b',',
        )
        .unwrap();
        let mut tree =
            RegressionTreeBuilder::new().strategy(TrainTestSplitStrategy::TrainTest(0.7));
        tree.fit(&dataset, "MEDV");
        let rmse = tree.evaluate(root_mean_square_error);
        dbg!(rmse);
    }
}
