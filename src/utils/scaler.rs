use ndarray::ArrayViewMut2;

use super::*;
use crate::{base_array::base_dataset::BaseDataset, dtype::DType, *};

#[derive(Clone, Default)]
pub enum ScalerState {
    #[default]
    None,
    MinMax,
    ZScore,
}

#[derive(Default)]
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
    pub fn fit(&mut self, data: &ArrayView2<f64>) {
        match self.state {
            ScalerState::None => {},
            ScalerState::MinMax => {
                let mut mins: Vec<f64> = vec![];
                let mut maxes: Vec<f64> = vec![];
                for (_, col) in data.columns().into_iter().enumerate() {
                    mins.push(
                        col.map(|x| NotNan::<f64>::from_f64(*x).unwrap())
                            .iter()
                            .min()
                            .unwrap()
                            .to_f64()
                            .unwrap(),
                    );
                    maxes.push(
                        col.map(|x| NotNan::<f64>::from_f64(*x).unwrap())
                            .iter()
                            .max()
                            .unwrap()
                            .to_f64()
                            .unwrap(),
                    );
                }
                self.maxes_stds = maxes;
                self.mins_means = mins;
            }
            ScalerState::ZScore => {
                let mut stds: Vec<f64> = vec![];
                let mut means: Vec<f64> = vec![];
                for (_, col) in data.columns().into_iter().enumerate() {
                    stds.push(col.std(0f64));
                    means.push(col.mean().unwrap());
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
            ScalerState::None => {},
            ScalerState::MinMax => {
                for (index, mut col) in data.columns_mut().into_iter().enumerate() {
                    for elem in col.iter_mut() {
                        *elem = (&*elem - self.mins_means[index])
                            / (self.maxes_stds[index] - self.mins_means[index])
                    }
                }
            },
            ScalerState::ZScore => {
                for (index, mut col) in data.columns_mut().into_iter().enumerate() {
                    for elem in col.iter_mut() {
                        *elem = (&*elem - self.mins_means[index]) / self.maxes_stds[index]
                    }
                }
            }
        }
    }
}

impl From<&ScalerState> for Scaler {
    fn from(value: &ScalerState) -> Self {
        Self {
            state: value.clone(),
            mins_means: Vec::default(),
            maxes_stds: Vec::default(),
        }
    }
}
