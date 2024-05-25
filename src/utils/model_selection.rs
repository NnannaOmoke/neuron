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
    pub(crate) train: Array2<f64>,
    train_target: Array1<f64>,
    test: Array2<f64>,
    test_target: Array1<f64>,
    eval: Array2<f64>,
    eval_target: Array1<f64>
}

impl RTrainTestSplitStrategyData {
    pub fn new(strategy: TrainTestSplitStrategy, dataset: &BaseDataset, target: usize,) -> Self {
        let complete_array = dataset.into_f64_array_without_target(target);
        let target = dataset.get_col(target).map(|x| x.to_f64().unwrap()).to_owned();
        let mut indices = Vec::from_iter(0..dataset.len());
        let mut rngs = thread_rng();
        indices.shuffle(&mut rngs);
        //this should shuffle the indices, create an intermediate 2d array that we'll split based on the train-test-split strategy
        let mut train = Array2::from_elem((0, dataset.shape().1 - 1), 0f64);
        let mut test = Array2::from_elem((0, dataset.shape().1 - 1), 0f64);
        let mut eval = Array2::from_elem((0, dataset.shape().1 -1), 0f64);
        let mut train_target = Vec::new();
        let mut test_target = Vec::new();
        let mut eval_target = Vec::new();
        match strategy {
            TrainTestSplitStrategy::None => {
                for elem in indices {
                    train
                        .push_row(complete_array
                            .row(elem)
                            .map(|x| *x)
                            .view(),)
                        .expect("Shape error");
                    train_target.push(target[elem]);
                }
            }
            TrainTestSplitStrategy::TrainTest(train_r) => {
                let train_ratio = (train_r * dataset.len() as f64) as usize;
                for elem in &indices[..train_ratio] {
                    train
                        .push_row(complete_array
                            .row(*elem)
                            .map(|x| *x)
                            .view())
                        .expect("Shape error");
                    train_target.push(target[*elem]);
                }
                
                for elem in &indices[train_ratio..] {
                    test.push_row(complete_array
                        .row(*elem)
                        .map(|x| *x)
                        .view())
                        .expect("Shape Error");
                    test_target.push(target[*elem]);
                }
            }
            TrainTestSplitStrategy::TrainTestEval(train_r, test_r, _) => {
                let train_ratio = (train_r * dataset.len() as f64) as usize;
                let test_ratio = (test_r * dataset.len() as f64) as usize;
                for elem in &indices[..train_ratio] {
                    train
                        .push_row(complete_array
                            .row(*elem)
                            .map(|x| *x)
                            .view())
                        .expect("Shape Error");
                    train_target.push(target[*elem]);
                }
                for elem in &indices[train_ratio..test_ratio + train_ratio] {
                    test.push_row(complete_array
                        .row(*elem)
                        .map(|x| *x)
                        .view())
                        .expect("Shape Error");
                    test_target.push(target[*elem]);
                }
                for elem in &indices[train_ratio + test_ratio..] {
                    eval.push_row(complete_array
                        .row(*elem)
                        .map(|x| *x)
                        .view())
                        .expect("Shape Error");
                    eval_target.push(target[*elem]);
                }
            }
        }
        drop(complete_array);
        Self {
            strategy,
            train,
            train_target: train_target.into(),
            test,
            test_target: test_target.into(),
            eval,
            eval_target: eval_target.into()
        }
    }
    pub fn get_train(&self) -> (ArrayView2<f64>, ArrayView1<f64>) {
        (self.train.view(), self.train_target.view())
    }

    pub fn get_train_mut(&mut self) -> (ArrayViewMut2<f64>, ArrayViewMut1<f64>) {
        (self.train.view_mut(), self.train_target.view_mut())
    }

    pub fn get_test(&self) -> (ArrayView2<f64>, ArrayView1<f64>) {
        (self.test.view(), self.test_target.view())
    }

    pub fn get_test_mut(&mut self) -> (ArrayViewMut2<f64>, ArrayViewMut1<f64>) {
        (self.test.view_mut(), self.test_target.view_mut())
    }

    pub fn get_eval(&self) -> (ArrayView2<f64>, ArrayView1<f64>) {
        (self.eval.view(), self.eval_target.view())
    }

    pub fn get_eval_mut(&mut self) -> (ArrayViewMut2<f64>, ArrayViewMut1<f64>) {
        (self.eval.view_mut(), self.eval_target.view_mut())
    }
   
}

impl Default for RTrainTestSplitStrategyData {
    fn default() -> Self {
        Self {
            strategy: TrainTestSplitStrategy::default(),
            train: Array2::default((0, 0)),
            train_target: Array1::default(0),
            test: Array2::default((0, 0)),
            test_target: Array1::default(0),
            eval: Array2::default((0, 0)),
            eval_target: Array1::default(0),
        }
    }
}

//tts for classification
pub struct CTrainTestSplitStrategyData {
    strategy: TrainTestSplitStrategy,
    pub(crate) train_features: Array2<f64>,
    train_target: Array1<u32>,
    test_features: Array2<f64>,
    test_target: Array1<u32>,
    eval_features: Array2<f64>,
    eval_target: Array1<u32>,
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
        let mut train_features = Array2::from_elem((0, dataset.shape().1 - 1), 0f64);
        let mut test_features = Array2::from_elem((0, dataset.shape().1 - 1), 0f64);
        let mut eval_features = Array2::from_elem((0, dataset.shape().1 - 1), 0f64);
        let mut train_target = Vec::new();
        let mut test_target = Vec::new();
        let mut eval_target = Vec::new();
        match strategy {
            TrainTestSplitStrategy::None => {
                for elem in indices {
                    train_features
                        .push_row(
                            feature_array
                                .row(elem)
                                .map(|x| *x)
                                .view(),
                        )
                        .expect("Shape error");
                    train_target.push(*target_arr.get(elem).unwrap());
                }
            }
            TrainTestSplitStrategy::TrainTest(train_r) => {
                let train_ratio = (train_r * dataset.len() as f64) as usize;
                for elem in &indices[..train_ratio] {
                    train_features
                        .push_row(
                            feature_array
                                .row(*elem)
                                .map(|x| *x)
                                .view(),
                        )
                        .expect("Shape error");
                    train_target.push(*target_arr.get(*elem).unwrap());
                }
                for elem in &indices[train_ratio..] {
                    test_features
                        .push_row(
                            feature_array
                                .row(*elem)
                                .map(|x| *x)
                                .view(),
                        )
                        .expect("Shape error");
                    test_target.push(*target_arr.get(target).unwrap());
                }
            }
            TrainTestSplitStrategy::TrainTestEval(train_r, test_r, _) => {
                let train_ratio = (train_r * dataset.len() as f64) as usize;
                let test_ratio = (test_r * dataset.len() as f64) as usize;
                for elem in &indices[..train_ratio] {
                    train_features
                        .push_row(
                            feature_array
                                .row(*elem)
                                .map(|x| *x)
                                .view(),
                        )
                        .expect("Shape error");
                    train_target.push(*target_arr.get(target).unwrap());
                }
                for elem in &indices[train_ratio..test_ratio + train_ratio] {
                    test_features
                        .push_row(
                            feature_array
                                .row(*elem)
                                .map(|x| *x)
                                .view(),
                        )
                        .expect("Shape error");
                    test_target.push(*target_arr.get(target).unwrap());
                }
                for elem in &indices[train_ratio + test_ratio..] {
                    eval_features
                        .push_row(
                            feature_array
                                .row(*elem)
                                .map(|x| *x)
                                .view(),
                        )
                        .expect("Shape error");
                    eval_target.push(*target_arr.get(target).unwrap());
                }
            }
        }
        drop(feature_array);
        Self {
            strategy,
            train_features,
            train_target: train_target.into(),
            test_features,
            test_target: test_target.into(),
            eval_features,
            eval_target: eval_target.into(),
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

    pub fn get_eval(&self) -> (ArrayView2<f64>, ArrayView1<u32>) {
        (self.eval_features.view(), self.eval_target.view())
    }

    pub fn get_eval_mut(&mut self) -> (ArrayViewMut2<f64>, ArrayViewMut1<u32>) {
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
