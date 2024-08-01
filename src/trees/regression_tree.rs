use core::{f64, num};
use std::collections::HashMap;

use naga::proc::index;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::{
    base_array::BaseDataset,
    utils::model_selection::{TrainTestSplitStrategy, TrainTestSplitStrategyData},
};

struct RegressionTreeNode {
    left: Option<Box<RegressionTreeNode>>,
    right: Option<Box<RegressionTreeNode>>,
    feature_idx: usize,
    gain: f64,
    value: f64,
    threshold: f64,
}
pub struct RegressionTreeBuilder {
    min_samples: usize,
    max_depth: usize,
    target_col: usize,
    strategy: TrainTestSplitStrategy,
    strategy_data: TrainTestSplitStrategyData<f64, f64>,
    root: Option<Box<RegressionTreeNode>>,
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
            min_samples: 10,
            max_depth: 10,
            target_col: 0,
            strategy: TrainTestSplitStrategy::None,
            strategy_data: TrainTestSplitStrategyData::default(),
            root: None,
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
            if *dataset.get((0, feature_idx)).unwrap() <= threshold {
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
    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target_col = dataset._get_string_index(target);
        self.strategy_data =
            TrainTestSplitStrategyData::<f64, f64>::new_r(dataset, self.target_col, self.strategy);
        let n = self.strategy_data.get_train().0.nrows();
        self.build_internal(Vec::from_iter(0..n), 0);
    }
    fn build_internal(
        &mut self,
        index_list: Vec<usize>,
        current_depth: usize,
    ) -> Box<RegressionTreeNode> {
        let dataset = self.strategy_data.get_train();
        let num_samples = dataset.0.nrows();
        if num_samples >= self.min_samples && current_depth <= self.max_depth {
            let best_split = self.best_split(dataset.0, dataset.1, &index_list);
            if best_split.gain != 0.0 {
                let left = self.build_internal(best_split.left, current_depth + 1);
                let right = self.build_internal(best_split.right, current_depth + 1);
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
        let value = Self::leaf_value(Self::pick_values(dataset.1, &index_list));
        Box::new(RegressionTreeNode {
            left: None,
            right: None,
            feature_idx: usize::MAX,
            gain: f64::NAN,
            threshold: f64::NAN,
            value,
        })
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
}
impl RegressionTreeNode {
    fn predict(&self, data: ArrayView1<f64>) -> f64 {
        if self.left.is_some() && self.right.is_some() {
            if data[self.feature_idx] <= self.threshold {
                self.left.as_ref().unwrap().predict(data)
            } else {
                self.left.as_ref().unwrap().predict(data)
            }
        } else {
            self.value
        }
    }
}