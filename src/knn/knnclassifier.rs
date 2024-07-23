use super::*;
use crate::base_array::BaseDataset;
use crate::knn::VotingChoice;
use crate::utils::scaler::{Scaler, ScalerState};
use crate::{
    utils::model_selection::{TrainTestSplitStrategy, TrainTestSplitStrategyData},
    *,
};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
pub struct KNNClassifierBuilder<M: Metric<f64>> {
    data: TrainTestSplitStrategyData<f64, u32>,
    strategy: TrainTestSplitStrategy,
    scaler: ScalerState,
    voting: VotingChoice,
    target: usize,
    p: PhantomData<M>,
    n: usize,
}

pub struct KNNClassifier<M: Metric<f64>> {
    internal: BallTreeKNN<'static, M>,
    config: KNNClassifierBuilder<M>,
}

impl<M: Metric<f64>> KNNClassifierBuilder<M> {
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
    ) -> KNNClassifier<M> {
        self.target = dataset._get_string_index(target);
        self.data =
            TrainTestSplitStrategyData::<f64, u32>::new_c(dataset, self.target, self.strategy);
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
        KNNClassifier::<M>::new(self, metric)
    }
}

impl<M: Metric<f64>> KNNClassifier<M> {
    pub fn new(config: KNNClassifierBuilder<M>, metric: M) -> Self {
        let data = config.data.get_train().0;
        let tree = BallTreeKNN::new(data, metric);
        Self {
            internal: tree,
            config,
        }
    }
    pub fn predict(&self, data: ArrayView2<f64>) -> Array1<u32> {
        let n = self.config.n;
        let results = self.internal.query(data, n);
        let mut model_results = Array1::from_elem(data.nrows(), 0);
        //TODO: Fix this; it's broken
        let values = Array2::from_shape_fn((results.0.nrows(), n), |(x, y)| {
            self.config.data.get_train().1[results.0[(x, y)]]
        });
        match self.config.voting {
            VotingChoice::Uniform => {
                model_results
                    .iter_mut()
                    .enumerate()
                    .for_each(|(index, ptr)| {
                        let counter: Counter<&u32, usize> = Counter::from_iter(values.row(index));
                        let most_common = counter.most_common();
                        let value = most_common[0].0;
                        if most_common[0].1 == most_common[0].1 {
                            let val = self.internal.tree.query_nearest(&data.row(index)).0;
                            *ptr = self.config.data.get_train().1[val];
                        } else {
                            *ptr = *value;
                        }
                    });
            }
            VotingChoice::Distance => {
                model_results
                    .iter_mut()
                    .enumerate()
                    .for_each(|(index, ptr)| {
                        let row = values.row(index);
                        let distances = results.1.row(index);
                        //it is more numerically stable to use an iter for the below
                        let mut distances = distances.to_owned();
                        distances.map_inplace(|x| {
                            if *x == 0f64 {
                            } else {
                                *x = 1.0 / *x
                            }
                        });
                        let mut sum = distances.sum();
                        if sum == 0f64 {
                            sum = 0.001;
                        }
                        let distances = distances / sum;
                        //so basically, we have a metric for how the distances will contribute to the scoring
                        let mut map = HashMap::new();
                        zip(row, distances).for_each(|(label, distance)| {
                            if !map.contains_key(label) {
                                map.insert(label, distance);
                            } else {
                                let mutref = map.get_mut(label).unwrap();
                                *mutref = &*mutref + distance;
                            }
                        });
                        let tup = map
                            .iter()
                            .max_by_key(|(_, value)| NotNan::new(**value).unwrap())
                            .unwrap();
                        *ptr = **tup.0;
                    });
            }
            VotingChoice::Custom(func) => {
                model_results
                    .iter_mut()
                    .enumerate()
                    .for_each(|(index, ptr)| {
                        let row = values.row(index);
                        let distances = results.1.row(index);
                        *ptr = func(row.map(|&x| x as f64).view(), distances) as u32;
                    });
            }
        }
        model_results
    }

    pub fn evaluate<F: Fn(ArrayView1<u32>, ArrayView1<u32>) -> Vec<f64>>(
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
mod test {
    use super::*;
    use crate::utils::metrics::accuracy;
    use crate::*;
    #[test]
    fn test_convergence_knn_c() {
        let dataset = BaseDataset::from_csv(
            Path::new("src/base_array/test_data/diabetes.csv"),
            true,
            true,
            b',',
        );
        let dataset = dataset.unwrap();
        let knn = KNNClassifierBuilder::new()
            .scaler(ScalerState::MinMax)
            .strategy(TrainTestSplitStrategy::TrainTest(0.7))
            .voting(VotingChoice::Distance)
            .set_n(5);
        let knn = knn.fit(&dataset, "Outcome", 5, Distance::Euclidean);
        let accuracy = knn.evaluate(accuracy);
        dbg!(accuracy[0]);
    }

    #[test]
    fn test_convergence_knn_multi() {
        let dataset = BaseDataset::from_csv(
            Path::new("src/base_array/test_data/IRIS.csv"),
            true,
            true,
            b',',
        );
        let dataset = dataset.unwrap();
        let knn = KNNClassifierBuilder::new()
            .scaler(ScalerState::MinMax)
            .strategy(TrainTestSplitStrategy::TrainTest(0.7))
            .voting(VotingChoice::Distance)
            .set_n(5);
        let knn = knn.fit(&dataset, "species", 5, Distance::Manhattan);
        let accuracy = knn.evaluate(accuracy);
        dbg!(accuracy[0]);
    }
}
