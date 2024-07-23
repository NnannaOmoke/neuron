use core::num;
use std::collections::HashMap;

use naga::proc::index;
use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::{
    base_array::BaseDataset,
    utils::model_selection::{TrainTestSplitStrategy, TrainTestSplitStrategyData},
};

struct ClassificationTreeNode {
    left: Option<Box<ClassificationTreeNode>>,
    right: Option<Box<ClassificationTreeNode>>,
    feature_idx: usize,
    gain: f64,
    value: u32,
    threshold: f64,
}
pub struct ClassificationTreeBuilder {
    min_samples: usize,
    max_depth: usize,
    target_col: usize,
    strategy: TrainTestSplitStrategy,
    strategy_data: TrainTestSplitStrategyData<f64, u32>,
    root: Option<Box<ClassificationTreeNode>>,
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

impl ClassificationTreeBuilder {
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
    fn entropy(data: &[u32]) -> f64 {
        let mut entropy = 0.0;
        Self::for_each_unique(data, |val| {
            let example_count = data.iter().filter(|&n| *n == val).count();
            let p1 = example_count as f64 / data.len() as f64;
            entropy += -p1 * p1.log2();
        });
        entropy
    }
    fn information_gain(parent: &[u32], left: &[u32], right: &[u32]) -> f64 {
        let parent_entropy = Self::entropy(parent);
        let w_left = left.len() as f64 / parent.len() as f64;
        let w_right = right.len() as f64 / parent.len() as f64;
        let e_left = Self::entropy(left);
        let e_right = Self::entropy(right);
        let wieghted_entropy = w_left * e_left + w_right * e_right;
        parent_entropy - wieghted_entropy
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
        target: ArrayView1<u32>,
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

                    let info_gain = Self::information_gain(&y, &left_y, &right_y);
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
    fn leaf_value(y: Vec<u32>) -> u32 {
        let mut highest_occurence = (0u32, 0usize); //(value, occurences)
        Self::for_each_unique(&y, |val| {
            let count = y.iter().filter(|&n| *n == val).count();
            if count > highest_occurence.1 {
                highest_occurence = (val, count);
            }
        });
        highest_occurence.0
    }
    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target_col = dataset._get_string_index(target);
        self.strategy_data =
            TrainTestSplitStrategyData::<f64, u32>::new_c(dataset, self.target_col, self.strategy);
        let n = self.strategy_data.get_train().0.nrows();
        self.build_internal(Vec::from_iter(0..n), 0);
    }
    fn build_internal(
        &mut self,
        index_list: Vec<usize>,
        current_depth: usize,
    ) -> Box<ClassificationTreeNode> {
        let dataset = self.strategy_data.get_train();
        let num_samples = dataset.0.nrows();
        if num_samples >= self.min_samples && current_depth <= self.max_depth {
            let best_split = self.best_split(dataset.0, dataset.1, &index_list);
            if best_split.gain != 0.0 {
                let left = self.build_internal(best_split.left, current_depth + 1);
                let right = self.build_internal(best_split.right, current_depth + 1);
                return Box::new(ClassificationTreeNode {
                    left: Some(left),
                    right: Some(right),
                    feature_idx: best_split.feature_idx,
                    gain: best_split.gain,
                    threshold: best_split.threshold,
                    value: u32::MAX,
                });
            }
        }
        let value = Self::leaf_value(Self::pick_values(dataset.1, &index_list));
        Box::new(ClassificationTreeNode {
            left: None,
            right: None,
            feature_idx: usize::MAX,
            gain: f64::NAN,
            threshold: f64::NAN,
            value,
        })
    }
}
