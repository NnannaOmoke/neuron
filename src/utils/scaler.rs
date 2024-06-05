use ndarray::ArrayViewMut2;

use super::*;
use crate::{base_array::base_dataset::BaseDataset, dtype::DType, *};

#[derive(Copy, Clone, Default, Debug)]
pub enum ScalerState {
    #[default]
    None,
    MinMax,
    ZScore,
}

#[derive(Default, Debug)]
pub struct Scaler {
    state: ScalerState,
    mins_means: Vec<f64>,
    maxes_stds: Vec<f64>,
}

impl ScalerState {
    pub fn normalize_dataset(&self, dataset: &mut BaseDataset, targetcol: usize) {
        match self {
            ScalerState::None => {}
            ScalerState::MinMax => {
                let mins = dataset
                    .columns()
                    .iter()
                    .enumerate()
                    .filter(|x| x.0 != targetcol)
                    .map(|x| dataset.min(x.1))
                    .collect::<Vec<DType>>();
                let maxs = dataset
                    .columns()
                    .iter()
                    .enumerate()
                    .filter(|x| x.0 != targetcol)
                    .map(|x| dataset.max(x.1))
                    .collect::<Vec<DType>>();
                for (index, mut col) in dataset.cols_mut().into_iter().enumerate() {
                    if index == targetcol {
                        continue;
                    }
                    for elem in col.iter_mut() {
                        *elem = ((&*elem - &mins[index]) / (&maxs[index] - &mins[index])).clone()
                    }
                }
            }
            ScalerState::ZScore => {
                let means = dataset
                    .columns()
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != targetcol)
                    .map(|(_, col)| dataset.mean(col))
                    .collect::<Vec<DType>>();
                let stds = dataset
                    .columns()
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != targetcol)
                    .map(|(_, col)| dataset.std(col))
                    .collect::<Vec<DType>>();

                for (index, mut col) in dataset
                    .cols_mut()
                    .into_iter()
                    .enumerate()
                    .filter(|(index, _)| *index != targetcol)
                {
                    for elem in col.iter_mut() {
                        *elem = ((&*elem - &means[index]) / &stds[index]).clone();
                    }
                }
            }
        }
    }
}

impl Scaler {
    pub fn fit(&mut self, data: ArrayView2<f64>) {
        match self.state {
            ScalerState::None => {}
            ScalerState::MinMax => {
                let mut mins = vec![];
                let mut maxes = vec![];
                for col in data.columns() {
                    mins.push(
                        col.iter()
                            .map(|x| NotNan::new(*x).unwrap())
                            .min()
                            .unwrap()
                            .to_f64()
                            .unwrap(),
                    );
                    maxes.push(
                        col.iter()
                            .map(|x| NotNan::new(*x).unwrap())
                            .max()
                            .unwrap()
                            .to_f64()
                            .unwrap(),
                    )
                }
                self.mins_means = mins;
                self.maxes_stds = maxes;
            }
            ScalerState::ZScore => {
                let mut stds = vec![];
                let mut means = vec![];
                for col in data.columns() {
                    stds.push(col.std(1.0));
                    means.push(col.mean().unwrap())
                }
                self.mins_means = means;
                self.maxes_stds = stds;
            }
        }
    }

    pub fn transform(&self, data: &mut ArrayViewMut2<f64>) {
        //should not be called without fitting!
        //assert!(self.maxes_stds.len() != 0);
        match self.state {
            ScalerState::None => {}
            ScalerState::MinMax => {
                for (index, mut col) in data.columns_mut().into_iter().enumerate() {
                    for elem in col.iter_mut() {
                        *elem = (*elem - self.mins_means[index])
                            / (self.maxes_stds[index] - self.mins_means[index])
                    }
                }
            }
            ScalerState::ZScore => {
                for (index, mut col) in data.columns_mut().into_iter().enumerate() {
                    for elem in col.iter_mut() {
                        *elem = (*elem - self.mins_means[index]) / self.maxes_stds[index]
                    }
                }
            }
        }
    }
}

impl From<&ScalerState> for Scaler {
    fn from(value: &ScalerState) -> Self {
        Self {
            state: *value,
            mins_means: Vec::default(),
            maxes_stds: Vec::default(),
        }
    }
}
