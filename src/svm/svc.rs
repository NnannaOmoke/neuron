use std::collections::btree_map::Values;

use crate::{
    base_array::BaseDataset,
    svm::{linear_kernel_2d, polynomial_kernel_2d, rbf_kernel_2d, sigmoid_kernel_2d, SVMKernel},
    utils::model_selection::{TrainTestSplitStrategy, TrainTestSplitStrategyData},
    utils::scaler::{Scaler, ScalerState},
    Array2, ArrayView2,
};

use ndarray::{array, Array1, ArrayView1};
use num_traits::ToPrimitive;
use prettytable::format::consts::FORMAT_NO_LINESEP_WITH_TITLE;

#[derive(Clone)]
pub struct SVCBuilder {
    C: f64,
    kkt_value: f64,
    kernel: SVMKernel,
    bias: f64,
    alphas: Array1<f64>,
    support_vectors: Array2<f64>,
    support_labels: Array1<u32>,
    strategy: TrainTestSplitStrategy,
    data: TrainTestSplitStrategyData<f64, u32>,
    target_index: usize,
    nclasses: usize,
    scaler: ScalerState,
}

impl SVCBuilder {
    pub fn new() -> Self {
        Self {
            C: 0.0,
            kkt_value: 1e-3, //reasonable default
            kernel: SVMKernel::Linear,
            strategy: TrainTestSplitStrategy::None,
            data: TrainTestSplitStrategyData::default(),
            target_index: 0,
            nclasses: 0,
            bias: 0.0,
            alphas: Array1::default(0),
            support_vectors: Array2::default((0, 0)),
            support_labels: Array1::default(0),
            scaler: ScalerState::default(),
        }
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn support_vectors(&self) -> (ArrayView2<f64>, ArrayView1<u32>) {
        (self.support_vectors.view(), self.support_labels.view())
    }

    pub fn alphas(&self) -> ArrayView1<f64> {
        self.alphas.view()
    }

    pub fn set_kkt(self, value: f64) -> Self {
        Self {
            kkt_value: value,
            ..self
        }
    }

    pub fn scaler(self, scaler: ScalerState) -> Self {
        Self { scaler, ..self }
    }

    pub fn train_test_split_strategy(self, strategy: TrainTestSplitStrategy) -> Self {
        Self { strategy, ..self }
    }

    pub fn set_c(self, c: f64) -> Self {
        Self { C: c, ..self }
    }

    pub fn kernel(self, kernel: SVMKernel) -> Self {
        Self { kernel, ..self }
    }

    //TODO: put this in a trait, because ATP it's boilerplate
    pub fn fit(&mut self, dataset: &BaseDataset, target: &str) {
        self.target_index = dataset._get_string_index(target);
        self.nclasses = dataset.nunique(target);
        self.data = TrainTestSplitStrategyData::<f64, u32>::new_c(
            dataset,
            self.target_index,
            self.strategy,
        );
        let mut scaler = Scaler::from(&self.scaler);
        scaler.fit(self.data.get_train().0);
        scaler.transform(&mut self.data.get_train_mut().0);
        match self.strategy {
            TrainTestSplitStrategy::TrainTest(_) => {
                scaler.transform(&mut self.data.get_test_mut().0);
            }
            TrainTestSplitStrategy::TrainTestEval(_, _, _) => {
                scaler.transform(&mut self.data.get_test_mut().0);
                scaler.transform(&mut self.data.get_eval_mut().0);
            }
            TrainTestSplitStrategy::None => {}
        };
        self.internal_fit();
    }

    pub fn internal_fit(&mut self) {
        let (features, labels) = self.data.get_train();
        let (features, labels) = (features.to_owned(), labels.to_owned());
        let kernel_ref = &self.kernel;
        let information = match self.nclasses {
            1 => panic!("Not enough classes"),
            2 => self.binomial_fit(features, labels, 20, kernel_ref),
            other => Self::multiclass_fit(features.view(), labels.view(), 20, other, &self.kernel),
        };
    }

    pub fn binomial_fit(
        &mut self,
        features: Array2<f64>,
        target: Array1<u32>,
        epochs: usize,
        kernel: &SVMKernel,
    ) {
        let (nrows, ncols) = (features.nrows(), features.ncols());
        let target = target.map(|x| x.to_f64().unwrap()).to_owned();
        let mut bias = 0.0;
        let mut alphas: Array1<f64> = Array1::zeros(nrows);
        let mut support_labels = target.to_owned();
        let mut support_vectors = features.to_owned();
        let mut non_kkt_array = Array1::from_shape_fn(nrows, |x| x);
        let mut error_cache = self
            .predict(features)
            .map(|x| x.to_f64().unwrap())
            .to_owned()
            - target;
        for _ in 0..epochs {
            let i2 = self.heuristic_2(&mut non_kkt_array);
            if i2 == -1 {
                break;
            }
        }
    }

    pub fn multiclass_fit(
        features: ArrayView2<f64>,
        target: ArrayView1<u32>,
        epochs: usize,
        nclasses: usize,
        kernel: &SVMKernel,
    ) {
        todo!()
    }

    pub fn predict(&self, input: ArrayView2<f64>) -> Array1<u32> {
        let weights = self.support_vectors.clone() * self.alphas.view();
        let values = match self.kernel {
            SVMKernel::RBF(gamma) => {
                rbf_kernel_2d(self.support_vectors.to_owned(), input.to_owned(), gamma)
            }
            SVMKernel::Polynomial(degree, coef_) => {
                polynomial_kernel_2d(self.support_vectors.view(), input.view(), degree, coef_)
            }
            SVMKernel::Sigmoid(coef_, alpha) => {
                sigmoid_kernel_2d(self.support_vectors.view(), input.view(), coef_, alpha)
            }
            SVMKernel::Linear => linear_kernel_2d(self.support_vectors.view(), input.view()),
        };

        let scores = weights.dot(&values.view()) + self.bias;
        todo!()
    }

    pub fn heuristic_1(&self, mut error_cache: &mut Array1<f64>, i2: isize) -> isize {
        let e2 = error_cache[i2 as usize];
        todo!()
    }

    pub fn heuristic_2(&mut self, mut non_kkt_array: &mut Array1<usize>) -> isize {
        let mut init_i2 = -1isize;
        for elem in non_kkt_array.clone() {
            non_kkt_array.remove_index(ndarray::Axis(0), elem);
            if !self.check_kkt(elem) {
                init_i2 = elem as isize;
                break;
            }
        }
        if init_i2 == -1 {
            //all samples satisfy KKT conditions, so we build new kKT array
            let indices = Array1::from_shape_fn(self.alphas.len(), |x| x);
            let kkt_indexes = self.check_kkt_multi(indices.view());
            let not_kkt_array = indices
                .iter()
                .filter(|x| !kkt_indexes[**x])
                .map(|x| *x)
                .collect::<Array1<usize>>();
            if non_kkt_array.len() > 0 {
                //TODO: shuffle
                //still stuff to do
                init_i2 = non_kkt_array[0] as isize;
                *non_kkt_array = non_kkt_array.slice(ndarray::s![1..-1]).to_owned();
            }
        }
        init_i2
    }

    pub fn check_kkt(&self, index: usize) -> bool {
        let alpha_index = self.alphas[index];
        let row = self.support_vectors.row(index);
        let mut row_2d = Array2::zeros((1, row.len()));
        row_2d.row_mut(1).assign(&row.view());
        let score = self.predict(row_2d.view())[0] as f64;
        let label_index = self.support_labels[index];
        let residual = label_index as f64 * score - 1f64;
        let condition_one = (alpha_index < self.C) && (residual < -self.kkt_value);
        let condition_two = (alpha_index > 0f64) && (residual > self.kkt_value);
        !(condition_one | condition_two)
    }

    fn check_kkt_multi(&self, indices: ArrayView1<usize>) -> Array1<bool> {
        let alphas = Array1::from_shape_fn(indices.len(), |x| self.alphas[indices[x]]);
        let array =
            Array2::from_shape_fn((alphas.len(), self.support_vectors.ncols()), |(x, y)| {
                self.support_vectors[(indices[x], y)]
            });
        let scores = self.predict(array.view()).map(|x| x.to_f64().unwrap());
        let label_indices = Array1::from_shape_fn(indices.len(), |x| {
            self.support_labels[indices[x]].to_f64().unwrap()
        });
        let residuals = label_indices * scores - 1f64;
        let condition_one = Array1::from_shape_fn(alphas.len(), |x| {
            (alphas[x] < self.C) && (residuals[x] < -self.kkt_value)
        });
        let condition_two = Array1::from_shape_fn(alphas.len(), |x| {
            (alphas[x] > 0f64) && (residuals[x] > self.kkt_value)
        });
        let res = Array1::from_shape_fn(alphas.len(), |x| !(condition_one[x] | condition_two[x]));
        res
    }
}
