use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayView2, ArrayView1, ArrayViewMut2, ArrayViewMut1};
use rand::{thread_rng, prelude::SliceRandom};
use crate::{base_array::base_dataset::BaseDataset, dtype::DTypeType};

#[derive(Copy, Clone, Default)]
pub enum TrainTestSplitStrategy {
    #[default]
    None,
    TrainTest(f64),
    TrainTestEval(f64, f64, f64),
}

//tts for regression
pub struct RTrainTestSplitStrategyData{
    strategy: TrainTestSplitStrategy,
    pub train: Array2<f64>,
    pub test: Array2<f64>,
    pub eval: Array2<f64>
}

impl RTrainTestSplitStrategyData{
    pub fn new(strategy: TrainTestSplitStrategy, dataset: &BaseDataset) -> Self{
        let complete_array = dataset.into_f64_array();
        let mut indices = Vec::from_iter(0..dataset.len());
        let mut rngs = thread_rng();
        indices.shuffle(&mut rngs);
        //this should shuffle the indices, create an intermediate 2d array that we'll split based on the train-test-split strategy
        let mut train = Array2::from_elem((0, dataset.shape().1), 0f64);
        let mut test = Array2::from_elem((0, dataset.shape().1), 0f64);
        let mut eval = Array2::from_elem((0, dataset.shape().1), 0f64);
        match strategy{
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
        Self{
            strategy, train, test, eval
        }
    }

    pub fn get_train(&self) -> ArrayView2<f64>{
        self.train.view()
    }

    pub fn get_train_mut(&mut self) -> ArrayViewMut2<f64>{
        self.train.view_mut()
    }

    pub fn get_test(&self) -> ArrayView2<f64>{
        self.test.view()
    }

    pub fn get_test_mut(&mut self) -> ArrayViewMut2<f64>{
        self.test.view_mut()
    }

    pub fn get_eval(&self) -> ArrayView2<f64>{
        self.eval.view()
    }

    pub fn get_eval_mut(&mut self) -> ArrayViewMut2<f64>{
        self.eval.view_mut()
    }
}

impl Default for RTrainTestSplitStrategyData{
    fn default() -> Self {
        Self{
            strategy: TrainTestSplitStrategy::default(),
            train: Array2::default((0, 0)),
            test: Array2::default((0, 0)),
            eval: Array2::default((0, 0))
        }
    }
}


//tts for classification 
pub struct CTrainTestSplitStrategyData{
    strategy: TrainTestSplitStrategy,
    mappings: Option<HashMap<String, u32>>,
    pub train_features: Array2<f64>,
    pub train_target: Array1<u32>,
    pub test_features: Array2<f64>,
    pub test_target: Array1<u32>,
    pub eval_features: Array2<f64>,
    pub eval_target: Array1<u32>,
}

impl CTrainTestSplitStrategyData{
    pub fn new(strategy: TrainTestSplitStrategy, dataset: &BaseDataset, target: usize) -> Self{
        let feature_array = dataset.into_f64_array_without_target(target);
        let dtype = dataset.get_col(target).first().unwrap().data_type();
        todo!();
    }

    pub fn init_mappings(target: ArrayView1<Box<String>>) -> Option<HashMap<String, i32>>{
        let mut map = HashMap::new();
        let mut assigned = 0;
        target.iter().for_each(|x|{
                if !map.contains_key(x.as_ref()){
                    map.entry(*x.clone()).or_insert(assigned);
                    assigned += 1;
                }
            } 
        );
        Some(map)
    }
}

impl Default for CTrainTestSplitStrategyData{
    fn default() -> Self {
        Self{
            strategy:TrainTestSplitStrategy::default(),
            mappings: Option::default(),
            train_features: Array2::default((0, 0)),
            train_target: Array1::default(0),
            test_features: Array2::default((0, 0)),
            test_target: Array1::default(0),
            eval_features: Array2::default((0, 0)),
            eval_target: Array1::default(0),
        }
    }
}
