use std::{arch::x86_64, collections::HashMap};

use crate::{
    base_array::base_dataset::BaseDataset,
    dtype::{DType, DTypeType},
    Axis,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, RawDataSubst};
use num_traits::{ToPrimitive, Zero};
use rand::{distributions::Distribution, distributions::Uniform, prelude::SliceRandom, thread_rng};

#[derive(Copy, Clone, Default)]
pub enum TrainTestSplitStrategy {
    #[default]
    None,
    TrainTest(f64),
    TrainTestEval(f64, f64, f64),
}

#[derive(Clone)]
pub struct TrainTestSplitStrategyData<F: Default + Clone, L: Default + Clone> {
    pub(crate) train_features: Array2<F>,
    train_labels: Array1<L>,
    test_features: Option<Array2<F>>,
    test_labels: Option<Array1<L>>,
    eval_features: Option<Array2<F>>,
    eval_labels: Option<Array1<L>>,
    strategy: TrainTestSplitStrategy,
}

impl<F: Default + Clone, L: Default + Clone> TrainTestSplitStrategyData<F, L> {
    pub fn new_r(
        dataset: &BaseDataset,
        target_col: usize,
        strategy: TrainTestSplitStrategy,
    ) -> TrainTestSplitStrategyData<f64, f64> {
        let array = dataset.into_f64_array_without_target(target_col);
        let target = dataset.get_col(target_col).map(|x| x.to_f64().unwrap());
        match strategy {
            TrainTestSplitStrategy::None => TrainTestSplitStrategyData {
                train_features: array,
                train_labels: target,
                strategy,
                ..Default::default()
            },
            TrainTestSplitStrategy::TrainTest(value) => {
                let (train_features, train_labels, test_features, test_labels) =
                    train_test_split(array, target, (value as f32, 1.0 - value as f32));
                TrainTestSplitStrategyData {
                    train_features,
                    train_labels,
                    test_features: Some(test_features),
                    test_labels: Some(test_labels),
                    strategy,
                    ..Default::default()
                }
            }
            TrainTestSplitStrategy::TrainTestEval(train, test, eval) => {
                let (train_features, train_labels, test_features, test_labels) =
                    train_test_split(array, target, (train as f32, 1.0 - test as f32));
                //calculate eval to test-ratio
                let test_eval_ratio = test / (test + eval);
                let (test_features, test_labels, eval_features, eval_labels) = train_test_split(
                    test_features,
                    test_labels,
                    (test_eval_ratio as f32, 1.0 - test_eval_ratio as f32),
                );
                TrainTestSplitStrategyData {
                    train_features,
                    train_labels,
                    test_features: Some(test_features),
                    test_labels: Some(test_labels),
                    eval_features: Some(eval_features),
                    eval_labels: Some(eval_labels),
                    strategy,
                }
            }
        }
    }

    pub fn new_c(
        dataset: &BaseDataset,
        target_col: usize,
        strategy: TrainTestSplitStrategy,
    ) -> TrainTestSplitStrategyData<f64, u32> {
        let array = dataset.into_f64_array_without_target(target_col);
        let target = dataset.get_col(target_col);
        let target = TrainTestSplitStrategyData::<f64, u32>::preprocess_c(target.to_owned());
        match strategy {
            TrainTestSplitStrategy::None => {
                return TrainTestSplitStrategyData {
                    train_features: array,
                    train_labels: target,
                    strategy,
                    ..Default::default()
                }
            }
            TrainTestSplitStrategy::TrainTest(value) => {
                let (train_features, train_labels, test_features, test_labels) =
                    train_test_split(array, target, (value as f32, 1.0 - value as f32));
                return TrainTestSplitStrategyData {
                    train_features,
                    train_labels,
                    test_features: Some(test_features),
                    test_labels: Some(test_labels),
                    strategy,
                    ..Default::default()
                };
            }
            TrainTestSplitStrategy::TrainTestEval(train, test, eval) => {
                let (train_features, train_labels, test_features, test_labels) =
                    train_test_split(array, target, (train as f32, 1.0 - test as f32));
                //calculate eval to test-ratio
                let test_eval_ratio = test / (test + eval);
                let (test_features, test_labels, eval_features, eval_labels) = train_test_split(
                    test_features,
                    test_labels,
                    (test_eval_ratio as f32, 1.0 - test_eval_ratio as f32),
                );
                return TrainTestSplitStrategyData {
                    train_features,
                    train_labels,
                    test_features: Some(test_features),
                    test_labels: Some(test_labels),
                    eval_features: Some(eval_features),
                    eval_labels: Some(eval_labels),
                    strategy,
                };
            }
        }
    }

    fn preprocess_c(column: Array1<DType>) -> Array1<u32> {
        fn map_string(target: ArrayView1<Box<String>>) -> Array1<u32> {
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
        let first = column.first().unwrap();
        let complete = match first {
            DType::Object(_) => map_string(
                column
                    .map(|x| match x {
                        DType::Object(inner) => inner.clone(),
                        _ => unreachable!(),
                    })
                    .view(),
            ),
            _ => column.map(|x| x.to_u32().unwrap()),
        };
        complete
    }

    pub fn get_train(&self) -> (ArrayView2<F>, ArrayView1<L>) {
        (self.train_features.view(), self.train_labels.view())
    }

    pub fn get_train_mut(&mut self) -> (ArrayViewMut2<F>, ArrayViewMut1<L>) {
        (self.train_features.view_mut(), self.train_labels.view_mut())
    }

    pub fn get_test(&self) -> (ArrayView2<F>, ArrayView1<L>) {
        (
            self.test_features.as_ref().unwrap().view(),
            self.test_labels.as_ref().unwrap().view(),
        )
    }

    pub fn get_test_mut(&mut self) -> (ArrayViewMut2<F>, ArrayViewMut1<L>) {
        (
            self.test_features.as_mut().unwrap().view_mut(),
            self.test_labels.as_mut().unwrap().view_mut(),
        )
    }

    pub fn get_eval(&self) -> (ArrayView2<F>, ArrayView1<L>) {
        (
            self.eval_features.as_ref().unwrap().view(),
            self.eval_labels.as_ref().unwrap().view(),
        )
    }

    pub fn get_eval_mut(&mut self) -> (ArrayViewMut2<F>, ArrayViewMut1<L>) {
        (
            self.eval_features.as_mut().unwrap().view_mut(),
            self.eval_labels.as_mut().unwrap().view_mut(),
        )
    }
}

impl<T: Default + Clone, L: Default + Clone> Default for TrainTestSplitStrategyData<T, L> {
    fn default() -> Self {
        Self {
            train_features: Array2::default((0, 0)),
            train_labels: Array1::default(0),
            test_features: Option::default(),
            test_labels: Option::default(),
            eval_features: Option::default(),
            eval_labels: Option::default(),
            strategy: TrainTestSplitStrategy::default(),
        }
    }
}

pub fn train_test_split<F: Default + Clone, L: Clone + Default>(
    features: Array2<F>,
    labels: Array1<L>,
    ratio: (f32, f32),
) -> (Array2<F>, Array1<L>, Array2<F>, Array1<L>) {
    //get indices, assumes features and labels have the same length
    assert!(features.nrows() == labels.len());
    assert!(ratio.0 + ratio.1 == 1.0);
    let mut rngs = thread_rng();
    let mut indices = Vec::from_iter(0..features.nrows());
    indices.shuffle(&mut rngs);
    let train_split = (ratio.0 * (indices.len() as f32)) as usize;
    let test_split = features.nrows() - train_split;

    let mut feature_train = Array2::from_elem((0, features.ncols()), F::default());
    let mut label_train = Array1::from_elem(features.nrows() - test_split, L::default());
    let mut feature_test = Array2::from_elem((0, features.ncols()), F::default());
    let mut label_test = Array1::from_elem(test_split, L::default());
    let mut count_t = 0;
    let mut count_test = 0;

    for elem in &indices[..train_split] {
        feature_train.push_row(features.row(*elem)).unwrap();
        label_train[count_t] = labels[*elem].clone();
        count_t += 1;
    }

    for elem in &indices[train_split..] {
        feature_test.push_row(features.row(*elem)).unwrap();
        label_test[count_test] = labels[*elem].clone();
        count_test += 1;
    }
    (feature_train, label_train, feature_test, label_test)
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    #[test]
    fn tts() {
        let mut count = 0;
        let features = Array2::from_shape_fn((100, 10), |_| {
            count += 1;
            count
        });
        let mut count = 0;
        let target = Array1::from_shape_fn(100, |_| {
            count += 1;
            count
        });
        let (ft, lt, ftest, ltest) = train_test_split(features, target, (0.7, 0.3));
        dbg!(&ft, &lt, &ftest, &ltest);
    }
}
