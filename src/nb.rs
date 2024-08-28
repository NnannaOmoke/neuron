use crate::base_array::BaseDataset;
use crate::utils::math::argmax_1d_f64;
use crate::utils::model_selection::{TrainTestSplitStrategy, TrainTestSplitStrategyData};
use ndarray::prelude::*;
use std::fmt::Display;
use std::iter::zip;

pub struct MultinomialNB {
    data: TrainTestSplitStrategyData<f64, u32>,
    strategy: TrainTestSplitStrategy,
    nb: RawNB,
    target: usize,
}

#[derive(Clone, Default, Debug)]
pub struct RawNB {
    alpha: f64,
    weights: Array2<f64>,
    nclasses: usize,
}

impl RawNB {
    fn multinomial_fit(&mut self, features: ArrayView2<f64>, labels: ArrayView1<u32>) {
        let alpha = self.alpha;
        let nclasses = self.nclasses;
        let mut prob_per_class = Array2::zeros((features.ncols(), nclasses));
        let mut class_vocab_count = Array1::from_elem(nclasses, 0f64);
        (0..nclasses).for_each(|class| {
            let mut sum = 0f64;
            let indices = labels
                .iter()
                .enumerate()
                .filter(|(_, &curr_class)| curr_class == class as u32)
                .map(|(index, _)| index)
                .collect::<Vec<usize>>();
            indices
                .iter()
                .for_each(|&index| sum += features.row(index).sum());
            class_vocab_count[nclasses] = sum + features.ncols() as f64; //laplacian smoothing for alphas
        });
        features
            .columns()
            .into_iter()
            .enumerate()
            .for_each(|(index, feature)| {
                let mut row = Array1::from_elem(nclasses, 0f64);
                zip(feature, labels).for_each(|(current, &label)| row[label as usize] += current);
                row += alpha;
                row.iter_mut()
                    .enumerate()
                    .for_each(|(index, value)| *value = *value / class_vocab_count[index]);
                prob_per_class.row_mut(index).assign(&row);
            });
        self.weights = prob_per_class;
    }

    fn predict(&self, input: ArrayView2<usize>) -> Array1<u32> {
        assert_eq!(self.weights.ncols(), input.ncols());
        let mut return_val = Array1::from_elem(input.nrows(), 0u32);
        let mut res_array = Array2::from_elem((input.nrows(), self.nclasses), 0f64);
        input
            .rows()
            .into_iter()
            .enumerate()
            .for_each(|(index, row)| {
                let mut results = Array1::from_elem(self.nclasses, 0f64);
                row.iter()
                    .filter(|&&weight| weight != usize::MAX)
                    .for_each(|&value| {
                        (0..self.nclasses).for_each(|class| {
                            results[class] += self.weights.row(value)[class].log10()
                        });
                    });
                res_array.row_mut(index).assign(&results);
            });
        res_array
            .rows()
            .into_iter()
            .enumerate()
            .for_each(|(index, row)| return_val[index] = argmax_1d_f64(row) as u32);
        return_val
    }
}

impl MultinomialNB {
    fn new() -> Self {
        Self {
            data: TrainTestSplitStrategyData::default(),
            strategy: TrainTestSplitStrategy::default(),
            nb: RawNB::default(),
            target: 0,
        }
    }

    pub fn strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        Self { strategy, ..self }
    }

    pub fn alphas(self, alpha: f64) -> Self {
        let prev = self.nb;
        Self {
            nb: RawNB { alpha, ..prev },
            ..self
        }
    }

    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target = dataset._get_string_index(target);
        self.nb.nclasses = dataset.nunique(target);
        self.data =
            TrainTestSplitStrategyData::<f64, u32>::new_c(dataset, self.target, self.strategy);
        self.internal_fit();
    }

    fn internal_fit(&mut self) {
        let (features, labels) = self.data.get_train();
        self.nb.multinomial_fit(features, labels)
    }

    pub fn predict(&self, data: ArrayView2<usize>) -> Array1<u32> {
        self.nb.predict(data)
    }

    pub fn evaluate<F>(&self, function: F) -> Vec<f64>
    where
        F: Fn(ArrayView1<u32>, ArrayView1<u32>) -> Vec<f64>,
    {
        let (features, ground_truth) = match self.strategy {
            TrainTestSplitStrategy::None => {
                //get train data
                self.data.get_train()
            }
            TrainTestSplitStrategy::TrainTest(_) => self.data.get_test(),
            TrainTestSplitStrategy::TrainTestEval(_, _, _) => self.data.get_eval(),
        };
        //TODO: implement after writing vectorizers and tfidf
        unimplemented!()
    }
}

impl Display for MultinomialNB {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[{:?}]", self.nb)
    }
}
