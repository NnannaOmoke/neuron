use self::knnclassifier::RawKNNClassifier;

use super::*;
use crate::base_array::BaseDataset;
use crate::utils::model_selection::TrainTestSplitStrategy;
use crate::utils::model_selection::TrainTestSplitStrategyData;
use crate::utils::scaler::{Scaler, ScalerState};
use petal_neighbors::distance::Euclidean;
use std::marker::PhantomData;

#[derive(Clone)]
pub struct RawKNNRegressor<'lt, M: Metric<f64>> {
    tree: BallTreeKNN<'lt, M>,
    labels: Array1<f64>,
    voting: VotingChoice,
    distance: Distance,
    n: usize,
}

pub struct KNNRegressorBuilder<'lt> {
    tree: RawKNNRegressor<'lt, Distance>,
    data: TrainTestSplitStrategyData<f64, f64>,
    strategy: TrainTestSplitStrategy,
    scaler: ScalerState,
    target: usize,
}

impl Default for RawKNNRegressor<'_, Distance> {
    fn default() -> Self {
        Self {
            tree: BallTreeKNN::new(Array2::from_elem((1, 1), 1f64).view(), Distance::Euclidean),
            labels: Array1::default(0),
            voting: VotingChoice::default(),
            distance: Distance::default(),
            n: 0,
        }
    }
}

impl std::fmt::Debug for RawKNNRegressor<'_, Distance> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[{:?}, {:?}]", self.voting, self.distance)
    }
}

impl<'lt> RawKNNRegressor<'_, Distance> {
    fn fit(&mut self, features: ArrayView2<f64>, labels: ArrayView1<f64>) {
        let tree = BallTreeKNN::new(features, self.distance);
        self.labels = labels.to_owned();
        self.tree = tree;
    }

    fn predict(&self, data: ArrayView2<f64>) -> Array1<f64> {
        let n = self.n;
        let results = self.tree.query(data, n);
        let mut model_results = Array1::from_elem(data.nrows(), 0f64);
        let values = Array2::from_shape_fn((results.0.nrows(), n), |(x, y)| {
            self.labels[results.0[(x, y)]]
        });
        match self.voting {
            VotingChoice::Uniform => {
                model_results
                    .iter_mut()
                    .enumerate()
                    .for_each(|(index, ptr)| *ptr = values.row(index).mean().unwrap());
            }
            VotingChoice::Distance => {
                model_results
                    .iter_mut()
                    .enumerate()
                    .for_each(|(index, ptr)| {
                        let row = values.row(index);
                        let distances = results.1.row(index);
                        let distances = 1.0 / distances.to_owned();
                        let sum = distances.sum();
                        let distances = distances / sum;
                        *ptr = distances.dot(&row);
                    });
            }
            VotingChoice::Custom(func) => {
                model_results
                    .iter_mut()
                    .enumerate()
                    .for_each(|(index, ptr)| {
                        let row = values.row(index);
                        let distances = results.1.row(index);
                        *ptr = func(row, distances);
                    });
            }
        }
        model_results
    }
}

impl<'lt> KNNRegressorBuilder<'lt> {
    pub fn new() -> Self {
        Self {
            tree: RawKNNRegressor::default(),
            strategy: TrainTestSplitStrategy::default(),
            data: TrainTestSplitStrategyData::default(),
            scaler: ScalerState::default(),
            target: 0,
        }
    }
    pub fn scaler(self, scaler: ScalerState) -> Self {
        Self { scaler, ..self }
    }

    pub fn strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        Self { strategy, ..self }
    }

    pub fn voting(self, voting: VotingChoice) -> Self {
        let prev = self.tree;
        Self {
            tree: RawKNNRegressor { voting, ..prev },
            ..self
        }
    }

    pub fn set_n(self, n: usize) -> Self {
        let prev = self.tree;
        Self {
            tree: RawKNNRegressor { n, ..prev },
            ..self
        }
    }

    pub fn distance(self, distance: Distance) -> Self {
        let prev = self.tree;
        Self {
            tree: RawKNNRegressor { distance, ..prev },
            ..self
        }
    }

    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target = dataset._get_string_index(target);
        self.data =
            TrainTestSplitStrategyData::<f64, f64>::new_r(dataset, self.target, self.strategy);
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

    fn internal_fit(&mut self) {
        let (features, labels) = self.data.get_train();
        self.tree.fit(features, labels);
    }

    pub fn predict(&self, data: ArrayView2<f64>) -> Array1<f64> {
        self.tree.predict(data)
    }

    pub fn evaluate<F: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Vec<f64>>(
        &self,
        function: F,
    ) -> Vec<f64> {
        //placeholder
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

    pub fn raw_mut(&'lt mut self) -> &'lt mut RawKNNRegressor<Distance> {
        &mut self.tree
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::metrics::root_mean_square_error;
    #[test]
    fn test_convergence_knn_r() {
        let dataset = BaseDataset::from_csv(
            Path::new("src/base_array/test_data/boston.csv"),
            true,
            true,
            b',',
        )
        .unwrap();
        let mut builder = KNNRegressorBuilder::new()
            .voting(VotingChoice::Uniform)
            .set_n(5)
            .scaler(ScalerState::MinMax)
            .distance(Distance::Euclidean)
            .strategy(TrainTestSplitStrategy::TrainTest(0.7));
        builder.fit(&dataset, "MEDV");
        let value = builder.evaluate(root_mean_square_error)[0];
        dbg!(value);
    }
}
