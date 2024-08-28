use super::*;
use crate::base_array::BaseDataset;
use crate::knn::Distance;
use crate::knn::VotingChoice;
use crate::utils::scaler::{Scaler, ScalerState};
use crate::{
    utils::model_selection::{TrainTestSplitStrategy, TrainTestSplitStrategyData},
    *,
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone)]
pub struct RawKNNClassifier<'lt, M: Metric<f64>> {
    tree: BallTreeKNN<'lt, M>,
    labels: Array1<u32>,
    voting: VotingChoice,
    distance: Distance,
    n: usize,
}

impl Debug for RawKNNClassifier<'_, Distance> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[{:?}, {:?}]", self.voting, self.distance)
    }
}

impl Default for RawKNNClassifier<'_, Distance> {
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

impl RawKNNClassifier<'_, Distance> {
    fn fit(&mut self, features: ArrayView2<f64>, labels: ArrayView1<u32>) {
        let tree = BallTreeKNN::new(features, self.distance);
        self.labels = labels.to_owned();
        self.tree = tree;
    }

    fn predict(&self, data: ArrayView2<f64>) -> Array1<u32> {
        let n = self.n;
        let results = self.tree.query(data, n);
        let mut model_results = Array1::from_elem(data.nrows(), 0);
        let values = Array2::from_shape_fn((results.0.nrows(), n), |(x, y)| {
            self.labels[results.0[(x, y)]]
        });
        let tree = self.tree.tree.read().unwrap();
        match self.voting {
            VotingChoice::Uniform => {
                model_results
                    .iter_mut()
                    .enumerate()
                    .for_each(|(index, ptr)| {
                        let counter: Counter<&u32, usize> = Counter::from_iter(values.row(index));
                        let most_common = counter.most_common();
                        let value = most_common[0].0;
                        if most_common[0].1 == most_common[0].1 {
                            let val = tree.query_nearest(&data.row(index)).0;
                            *ptr = self.labels[val];
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
}

pub struct KNNClassifierBuilder<'lt> {
    tree: RawKNNClassifier<'lt, Distance>,
    data: TrainTestSplitStrategyData<f64, u32>,
    strategy: TrainTestSplitStrategy,
    scaler: ScalerState,
    target: usize,
}

impl<'lt> KNNClassifierBuilder<'lt> {
    pub fn new() -> Self {
        Self {
            tree: RawKNNClassifier::default(),
            data: TrainTestSplitStrategyData::default(),
            strategy: TrainTestSplitStrategy::default(),
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
            tree: RawKNNClassifier { voting, ..prev },
            ..self
        }
    }

    pub fn set_n(self, n: usize) -> Self {
        let prev = self.tree;
        Self {
            tree: RawKNNClassifier { n, ..prev },
            ..self
        }
    }

    pub fn distance(self, distance: Distance) -> Self {
        let prev = self.tree;
        Self {
            tree: RawKNNClassifier { distance, ..prev },
            ..self
        }
    }

    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target = dataset._get_string_index(target);
        self.data =
            TrainTestSplitStrategyData::<f64, u32>::new_c(dataset, self.target, self.strategy);
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

    pub fn predict(&self, data: ArrayView2<f64>) -> Array1<u32> {
        self.tree.predict(data)
    }

    pub fn evaluate<F: Fn(ArrayView1<u32>, ArrayView1<u32>) -> Vec<f64>>(
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

    pub fn raw_mut(&'lt mut self) -> &'lt mut RawKNNClassifier<Distance> {
        &mut self.tree
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
        let mut knn = KNNClassifierBuilder::new()
            .scaler(ScalerState::MinMax)
            .strategy(TrainTestSplitStrategy::TrainTest(0.7))
            .voting(VotingChoice::Distance)
            .distance(Distance::Euclidean)
            .set_n(5);
        knn.fit(&dataset, "Outcome");
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
        let mut knn = KNNClassifierBuilder::new()
            .scaler(ScalerState::MinMax)
            .strategy(TrainTestSplitStrategy::TrainTest(0.7))
            .voting(VotingChoice::Distance)
            .distance(Distance::Euclidean)
            .set_n(5);
        knn.fit(&dataset, "species");
        let accuracy = knn.evaluate(accuracy);
        dbg!(accuracy[0]);
    }
}
