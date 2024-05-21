use std::collections::HashMap;

use crate::{
    base_array::base_dataset::BaseDataset,
    dtype::{DType, DTypeType},
    Axis,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use num_traits::ToPrimitive;
use rand::{prelude::SliceRandom, thread_rng};

#[derive(Copy, Clone, Default)]
pub enum TrainTestSplitStrategy {
    #[default]
    None,
    TrainTest(f64),
    TrainTestEval(f64, f64, f64),
}

//tts for regression
pub struct RTrainTestSplitStrategyData {
    strategy: TrainTestSplitStrategy,
    pub train: Array2<f64>,
    pub test: Array2<f64>,
    pub eval: Array2<f64>,
}

impl RTrainTestSplitStrategyData {
    pub fn new(strategy: TrainTestSplitStrategy, dataset: &BaseDataset) -> Self {
        let complete_array = dataset.into_f64_array();
        let mut indices = Vec::from_iter(0..dataset.len());
        let mut rngs = thread_rng();
        indices.shuffle(&mut rngs);
        //this should shuffle the indices, create an intermediate 2d array that we'll split based on the train-test-split strategy
        let mut train = Array2::from_elem((0, dataset.shape().1), 0f64);
        let mut test = Array2::from_elem((0, dataset.shape().1), 0f64);
        let mut eval = Array2::from_elem((0, dataset.shape().1), 0f64);
        match strategy {
            TrainTestSplitStrategy::None => {
                for elem in indices {
                    train
                        .push_row(complete_array.row(elem))
                        .expect("Shape error");
                }
            }
            TrainTestSplitStrategy::TrainTest(train_r) => {
                let train_ratio = (train_r * dataset.len() as f64) as usize;
                for elem in &indices[..train_ratio] {
                    train
                        .push_row(complete_array.row(*elem))
                        .expect("Shape error");
                }
                for elem in &indices[train_ratio..] {
                    test.push_row(complete_array.row(*elem))
                        .expect("Shape Error");
                }
            }
            TrainTestSplitStrategy::TrainTestEval(train_r, test_r, _) => {
                let train_ratio = (train_r * dataset.len() as f64) as usize;
                let test_ratio = (test_r * dataset.len() as f64) as usize;
                for elem in &indices[..train_ratio] {
                    train
                        .push_row(complete_array.row(*elem))
                        .expect("Shape Error");
                }
                for elem in &indices[train_ratio..test_ratio + train_ratio] {
                    test.push_row(complete_array.row(*elem))
                        .expect("Shape Error");
                }
                for elem in &indices[train_ratio + test_ratio..] {
                    eval.push_row(complete_array.row(*elem))
                        .expect("Shape Error");
                }
            }
        }
        drop(complete_array);
        Self {
            strategy,
            train,
            test,
            eval,
        }
    }

    pub fn get_train(&self) -> ArrayView2<f64> {
        self.train.view()
    }

    pub fn get_train_mut(&mut self) -> ArrayViewMut2<f64> {
        self.train.view_mut()
    }

    pub fn get_test(&self) -> ArrayView2<f64> {
        self.test.view()
    }

    pub fn get_test_mut(&mut self) -> ArrayViewMut2<f64> {
        self.test.view_mut()
    }

    pub fn get_eval(&self) -> ArrayView2<f64> {
        self.eval.view()
    }

    pub fn get_eval_mut(&mut self) -> ArrayViewMut2<f64> {
        self.eval.view_mut()
    }
}

impl Default for RTrainTestSplitStrategyData {
    fn default() -> Self {
        Self {
            strategy: TrainTestSplitStrategy::default(),
            train: Array2::default((0, 0)),
            test: Array2::default((0, 0)),
            eval: Array2::default((0, 0)),
        }
    }
}

//tts for classification
pub struct CTrainTestSplitStrategyData {
    strategy: TrainTestSplitStrategy,
    pub train_features: Array2<f64>,
    pub train_target: Array1<u32>,
    pub test_features: Array2<f64>,
    pub test_target: Array1<u32>,
    pub eval_features: Array2<f64>,
    pub eval_target: Array1<u32>,
}

impl CTrainTestSplitStrategyData {
    pub fn new(strategy: TrainTestSplitStrategy, dataset: &BaseDataset, target: usize) -> Self {
        let feature_array = dataset.into_f64_array_without_target(target);
        let dtype = dataset.get_col(target).first().unwrap().data_type();
        let target_arr = match dtype {
            DTypeType::F32 | DTypeType::F64 | DTypeType::U32 | DTypeType::U64 => {
                Array1::from_iter(dataset.get_col(target).map(|x| x.to_u32().unwrap()))
            }
            DTypeType::Object => Self::init_string_mappings(
                dataset
                    .get_col(target)
                    .map(|x| Box::new(x.to_string()))
                    .view(),
            ),
            _ => panic!("None type detected in target column"),
        };
        let mut indices = Vec::from_iter(0..dataset.len());
        let mut rngs = thread_rng();
        indices.shuffle(&mut rngs);
        //this should shuffle the indices, create an intermediate 2d array that we'll split based on the train-test-split strategy
        let mut train_features = Array2::from_elem((0, dataset.shape().1), 0f64);
        let mut test_features = Array2::from_elem((0, dataset.shape().1), 0f64);
        let mut eval_features = Array2::from_elem((0, dataset.shape().1), 0f64);
        let mut train_target = Array1::from_elem(dataset.shape().0, 0);
        let mut test_target = Array1::from_elem(dataset.shape().0, 0);
        let mut eval_target = Array1::from_elem(dataset.shape().0, 0);
        let mut count = 0;
        match strategy {
            TrainTestSplitStrategy::None => {
                for elem in indices {
                    train_features
                        .push_row(
                            feature_array
                                .row(elem)
                                .iter()
                                .enumerate()
                                .filter(|(index, _)| *index != target)
                                .map(|(_, x)| *x)
                                .collect::<Array1<f64>>()
                                .view(),
                        )
                        .expect("Shape error");
                    train_target[count] = *target_arr.get(elem).unwrap() as u32;
                    count += 1;
                }
                //cheese to save memory
                test_target = Array1::default(0);
                eval_target = Array1::default(0);
            }
            TrainTestSplitStrategy::TrainTest(train_r) => {
                let train_ratio = (train_r * dataset.len() as f64) as usize;
                for elem in &indices[..train_ratio] {
                    train_features
                        .push_row(
                            feature_array
                                .row(*elem)
                                .iter()
                                .enumerate()
                                .filter(|(index, _)| *index != target)
                                .map(|(_, x)| *x)
                                .collect::<Array1<f64>>()
                                .view(),
                        )
                        .expect("Shape error");
                    train_target[count] = *target_arr.get(*elem).unwrap();
                }
                count = 0;
                for elem in &indices[train_ratio..] {
                    test_features
                        .push_row(
                            feature_array
                                .row(*elem)
                                .iter()
                                .enumerate()
                                .filter(|(index, _)| *index != target)
                                .map(|(_, x)| *x)
                                .collect::<Array1<f64>>()
                                .view(),
                        )
                        .expect("Shape error");
                    test_target[count] = *target_arr.get(target).unwrap();
                }
                eval_target = Array1::default(0);
            }
            TrainTestSplitStrategy::TrainTestEval(train_r, test_r, _) => {
                let train_ratio = (train_r * dataset.len() as f64) as usize;
                let test_ratio = (test_r * dataset.len() as f64) as usize;
                for elem in &indices[..train_ratio] {
                    train_features
                        .push_row(
                            feature_array
                                .row(*elem)
                                .iter()
                                .enumerate()
                                .filter(|(index, _)| *index != target)
                                .map(|(_, x)| *x)
                                .collect::<Array1<f64>>()
                                .view(),
                        )
                        .expect("Shape error");
                    train_target[count] = *target_arr.get(target).unwrap();
                }
                count = 0;
                for elem in &indices[train_ratio..test_ratio + train_ratio] {
                    test_features
                        .push_row(
                            feature_array
                                .row(*elem)
                                .iter()
                                .enumerate()
                                .filter(|(index, _)| *index != target)
                                .map(|(_, x)| *x)
                                .collect::<Array1<f64>>()
                                .view(),
                        )
                        .expect("Shape error");
                    test_target[count] = *target_arr.get(target).unwrap();
                }
                count = 0;
                for elem in &indices[train_ratio + test_ratio..] {
                    eval_features
                        .push_row(
                            feature_array
                                .row(*elem)
                                .iter()
                                .enumerate()
                                .filter(|(index, _)| *index != target)
                                .map(|(_, x)| *x)
                                .collect::<Array1<f64>>()
                                .view(),
                        )
                        .expect("Shape error");
                    eval_target[count] = *target_arr.get(target).unwrap();
                }
            }
        }
        drop(feature_array);
        Self {
            strategy,
            train_features,
            train_target,
            test_features,
            test_target,
            eval_features,
            eval_target,
        }
    }

    pub fn init_string_mappings(target: ArrayView1<Box<String>>) -> Array1<u32> {
        let mut map = HashMap::new();
        let mut assigned = 0u32;
        target.iter().for_each(|x| {
            if !map.contains_key(x.as_ref()) {
                map.entry(*x.clone()).or_insert(assigned);
                assigned += 1;
            }
        });
        let res = Array1::from_iter(target.iter().map(|x| *map.get(x.as_ref()).unwrap()));
        res
    }

    pub fn get_train(&self) -> (ArrayView2<f64>, ArrayView1<u32>) {
        (self.train_features.view(), self.train_target.view())
    }

    pub fn get_train_mut(&mut self) -> (ArrayViewMut2<f64>, ArrayViewMut1<u32>) {
        (self.train_features.view_mut(), self.train_target.view_mut())
    }

    pub fn get_test(&self) -> (ArrayView2<f64>, ArrayView1<u32>) {
        (self.test_features.view(), self.test_target.view())
    }

    pub fn get_test_mut(&mut self) -> (ArrayViewMut2<f64>, ArrayViewMut1<u32>) {
        (self.test_features.view_mut(), self.test_target.view_mut())
    }

    pub fn get_eval(&self) -> (ArrayView2<f64>,ArrayView1<u32>) {
        (self.eval_features.view(), self.eval_target.view())
    }

    pub fn get_eval_mut(&mut self) -> (ArrayViewMut2<f64>, ArrayViewMut1<u32>){
        (self.eval_features.view_mut(), self.eval_target.view_mut())
    }
}

impl Default for CTrainTestSplitStrategyData {
    fn default() -> Self {
        Self {
            strategy: TrainTestSplitStrategy::default(),
            train_features: Array2::default((0, 0)),
            train_target: Array1::default(0),
            test_features: Array2::default((0, 0)),
            test_target: Array1::default(0),
            eval_features: Array2::default((0, 0)),
            eval_target: Array1::default(0),
        }
    }
}
