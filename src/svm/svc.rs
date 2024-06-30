use crate::{
    base_array::BaseDataset,
    svm::SVMKernel,
    utils::model_selection::{TrainTestSplitStrategy, TrainTestSplitStrategyData},
    utils::scaler::{Scaler, ScalerState},
    Array2, ArrayView2,
};

use ndarray::{Array1, ArrayView1};

#[derive(Clone)]
pub struct SVCBuilder {
    C: f64,
    kkt_value: f64,
    kernel: SVMKernel,
    bias: Array1<f64>,
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
            bias: Array1::default(0),
            alphas: Array1::default(0),
            support_vectors: Array2::default((0, 0)),
            support_labels: Array1::default(0),
            scaler: ScalerState::default(),
        }
    }

    pub fn bias(&self) -> ArrayView1<f64> {
        self.bias.view()
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
        let information = match self.nclasses {
            1 => panic!("Not enough classes"),
            2 => Self::binomial_fit(features, labels, 20, &self.kernel),
            other => Self::multiclass_fit(features, labels, 20, other, &self.kernel),
        };
    }

    pub fn binomial_fit(
        features: ArrayView2<f64>,
        target: ArrayView1<u32>,
        epochs: usize,
        kernel: &SVMKernel,
    ) {
        todo!()
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
}
