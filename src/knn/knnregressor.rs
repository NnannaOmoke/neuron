use super::*;
use crate::base_array::BaseDataset;
use crate::utils::model_selection::TrainTestSplitStrategy;
use crate::utils::model_selection::TrainTestSplitStrategyData;
use crate::utils::scaler::{Scaler, ScalerState};
use petal_neighbors::distance::Euclidean;
use std::marker::PhantomData;

pub struct KNNRegressorBuilder<M: Metric<f64>> {
    data: TrainTestSplitStrategyData<f64, f64>,
    strategy: TrainTestSplitStrategy,
    scaler: ScalerState,
    voting: VotingChoice,
    target: usize,
    p: PhantomData<M>,
    n: usize,
}

pub struct KNNRegressor<M: Metric<f64>> {
    internal: BallTreeKNN<'static, M>,
    config: KNNRegressorBuilder<M>,
}

impl<M: Metric<f64>> KNNRegressorBuilder<M> {
    pub fn new() -> Self {
        Self {
            data: TrainTestSplitStrategyData::default(),
            strategy: TrainTestSplitStrategy::default(),
            scaler: ScalerState::default(),
            voting: VotingChoice::default(),
            target: 0,
            n: 0,
            p: PhantomData,
        }
    }

    pub fn set_n(self, n: usize) -> Self {
        Self { n, ..self }
    }
    pub fn scaler(self, scaler: ScalerState) -> Self {
        Self { scaler, ..self }
    }

    pub fn strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        Self { strategy, ..self }
    }

    pub fn voting(self, voting: VotingChoice) -> Self {
        Self { voting, ..self }
    }

    pub fn fit(
        mut self,
        dataset: &BaseDataset,
        target: &str,
        n: usize,
        metric: M,
    ) -> KNNRegressor<M> {
        self.target = dataset._get_string_index(target);
        self.data =
            TrainTestSplitStrategyData::<f64, f64>::new_r(dataset, self.target, self.strategy);
        let mut scaler = Scaler::from(&self.scaler);
        self.n = n;
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
        KNNRegressor::<M>::new(self, metric)
    }
}

impl<M: Metric<f64>> KNNRegressor<M> {
    pub fn new(config: KNNRegressorBuilder<M>, metric: M) -> Self {
        let data = config.data.get_train().0;
        let tree = BallTreeKNN::new(data, metric);
        Self {
            internal: tree,
            config,
        }
    }
    pub fn predict(&self, data: ArrayView2<f64>) -> Array1<f64> {
        let n = self.config.n;
        let results = self.internal.query(data, n);
        let mut model_results = Array1::from_elem(data.nrows(), 0f64);
        let values = Array2::from_shape_fn((results.0.nrows(), n), |(x, y)| {
            self.config.data.get_train().1[results.0[(x, y)]]
        });
        match self.config.voting {
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

    pub fn evaluate<F: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Vec<f64>>(
        &self,
        function: F,
    ) -> Vec<f64> {
        //placeholder
        let (features, ground_truth) = match self.config.strategy {
            TrainTestSplitStrategy::None => {
                //get train data
                self.config.data.get_train()
            }
            TrainTestSplitStrategy::TrainTest(_) => self.config.data.get_test(),
            TrainTestSplitStrategy::TrainTestEval(_, _, _) => self.config.data.get_eval(),
        };
        let preds = self.predict(features);
        function(ground_truth, preds.view())
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
        let builder = KNNRegressorBuilder::new()
            .voting(VotingChoice::Uniform)
            .set_n(5)
            .scaler(ScalerState::MinMax)
            .strategy(TrainTestSplitStrategy::TrainTest(0.7));
        let fitted = builder.fit(&dataset, "MEDV", 5, Distance::Euclidean);
        let value = fitted.evaluate(root_mean_square_error)[0];
        dbg!(value);
    }
}
