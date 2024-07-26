use crate::base_array::BaseDataset;
use crate::utils::math::argmax_1d_f64;
use crate::utils::model_selection::{TrainTestSplitStrategy, TrainTestSplitStrategyData};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use std::iter::zip;

pub struct MultinomialNB {
    data: TrainTestSplitStrategyData<f64, u32>,
    strategy: TrainTestSplitStrategy,
    alpha: f64,
    weights: Array2<f64>,
    target: usize,
    nclasses: usize,
}

impl MultinomialNB {
    fn new(alpha: f64) -> Self {
        Self {
            data: TrainTestSplitStrategyData::default(),
            strategy: TrainTestSplitStrategy::default(),
            weights: Array2::default((0, 0)),
            alpha,
            target: 0,
            nclasses: 0,
        }
    }

    fn strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        Self { strategy, ..self }
    }

    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target = dataset._get_string_index(target);
        self.nclasses = dataset.nunique(target);
        self.data =
            TrainTestSplitStrategyData::<f64, u32>::new_c(dataset, self.target, self.strategy);
        let (features, labels) = self.data.get_train();
        self.weights = Self::multinomial_fit(features, labels, self.nclasses, self.alpha);
    }

    fn multinomial_fit(
        features: ArrayView2<f64>,
        labels: ArrayView1<u32>,
        nclasses: usize,
        alpha: f64,
    ) -> Array2<f64> {
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
        prob_per_class
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
