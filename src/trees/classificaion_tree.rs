use core::num;
use std::collections::HashMap;

use ndarray::{Array2, ArrayView1};

use crate::base_array::BaseDataset;


struct ClassificationTreeNode {
    left: Option<Box<ClassificationTreeNode>>,
    right: Option<Box<ClassificationTreeNode>>,
    feature_idx: usize,
    gain: f64,
    value: f64,
    threshold: f64
}
pub struct ClassificationTreeBuilder {
    min_samples: usize,
    max_depth: usize,
    target_col: usize
}
struct Split {
    gain: f64, 
    threshold: f64,
    feature_idx: usize, 
    left: Vec<usize>, 
    right: Vec<usize>
}
impl Split {
    fn new() -> Self {
        Self {
            gain: 0.0,
            feature_idx: 0,
            threshold: 0.0,
            left: vec![],
            right: vec![] 
        }
    }
}


impl ClassificationTreeBuilder {
    fn new() -> Self {
        ClassificationTreeBuilder {
            min_samples: 10,
            max_depth: 10,
            target_col: 0
        }
    }
    fn split_data(&self, dataset: &Array2<f64>, indices: &Vec<usize>, feature_idx: usize, threshold: f64) -> (Vec<usize>, Vec<usize>) {
        let mut left = Vec::<usize>::new();
        let mut right = Vec::<usize>::new();
        for row in indices {
            if *dataset.get((0, self.array_index_from_feature(feature_idx))).unwrap() <= threshold {
                left.push(*row);
            } else {
                right.push(*row);
            }
        }
        (left, right)
    }
    fn entropy(data: &Vec<f64>) -> f64 {
        let mut entropy = 0.0;
        Self::for_each_unique(data, |val| {
            let example_count = data.iter().filter(|&n| *n == val).count();
            let p1 = example_count as f64 / data.len() as f64;
            entropy += -p1 * p1.log2();
        });
        entropy
    }
    fn information_gain(parent: &Vec<f64>, left: &Vec<f64>, right: &Vec<f64>) -> f64 {
        let parent_entropy = Self::entropy(parent);
        let w_left = left.len() as f64 / parent.len() as f64;
        let w_right = right.len() as f64 / parent.len() as f64;
        let e_left = Self::entropy(left);
        let e_right = Self::entropy(right);
        let wieghted_entropy = w_left * e_left + w_right * e_right;
        parent_entropy - wieghted_entropy
    }
    fn array_index_from_feature(&self, index: usize) -> usize{
        if index < self.target_col {
            index
        } else {
            index + 1
        }
    }
    fn for_each_unique(array: &Vec<f64>, mut func: impl FnMut(f64) -> ()) {
        let mut encountered: Vec<f64> = vec![];
        for val in array {
            if !encountered.contains(val) {
                func(*val);
                encountered.push(*val);
            }
        }
    }
    fn column_from_index_list(dataset: &Array2<f64>, index_list: &Vec<usize>, col_index: usize) -> Vec<f64> {
        let mut out_column = Vec::<f64>::new();
        let column = dataset.column(col_index);
        for index in index_list {
            out_column.push(column[*index]);
        }
        out_column
    }
    fn best_split(&self, dataset: &Array2<f64>, indices: &Vec<usize>) -> Split { 
        let num_features = dataset.ncols() - 1;
        let mut best_split = Split::new();
        for feature_idx in 0..num_features {
            let feature = Self::column_from_index_list(dataset, &indices, self.array_index_from_feature(feature_idx));
            Self::for_each_unique(&feature, |val| {
                let (left, right) = self.split_data(dataset, indices, feature_idx, val);
                if !left.is_empty() && !right.is_empty() {
                    let y = Self::column_from_index_list(dataset, indices, self.target_col);
                    let left_y = Self::column_from_index_list(dataset, &left, self.target_col);
                    let right_y = Self::column_from_index_list(dataset, &right, self.target_col);

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
    pub fn build(&mut self, dataset: &BaseDataset, target: &str) {
        self.target_col = dataset._get_string_index(target);
        let features = dataset.into_f64_array();
        let n = features.nrows();
        self.build_internal(&features, Vec::from_iter(0..n), 0);
    }
    pub fn leaf_value(y: Vec<f64>) -> f64{
        let mut highest_occurence = (0.0, 0usize); //(value, occurences)
        Self::for_each_unique(&y, |val| {
            let count = y.iter().filter(|&n| *n == val).count();
            if count > highest_occurence.1 {
                highest_occurence = (val, count);
            }
        });
        highest_occurence.0
    }
    fn build_internal(&mut self, dataset: &Array2<f64>, index_list: Vec<usize>, current_depth: usize) -> Box<ClassificationTreeNode> {
        let num_samples = dataset.nrows();
        if num_samples >= self.min_samples && current_depth <= self.max_depth {
            let best_split = self.best_split(&dataset, &index_list);
            if best_split.gain != 0.0 {
                let left = self.build_internal(dataset, best_split.left, current_depth + 1);
                let right = self.build_internal(dataset, best_split.right, current_depth + 1);
                return Box::new(ClassificationTreeNode {
                    left: Some(left),
                    right: Some(right),
                    feature_idx: best_split.feature_idx,
                    gain: best_split.gain,
                    threshold: best_split.threshold,
                    value: f64::NAN
                })
            }
        } 
        let value = Self::leaf_value(Self::column_from_index_list(&dataset, &index_list, self.target_col));
        Box::new(ClassificationTreeNode{
            left: None,
            right: None,
            feature_idx: usize::MAX,
            gain: f64::NAN,
            threshold: f64::NAN,
            value
        })
    }
}   