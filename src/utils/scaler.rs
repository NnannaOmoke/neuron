use super::*;
use crate::{base_array::base_dataset::BaseDataset, dtype::DType, *};

pub enum Scaler {
    MinMax,
    ZScore,
}

pub fn normalize(dataset: &mut BaseDataset, standardizer: Scaler, targetcol: usize) {
    match standardizer {
        Scaler::MinMax => {
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
                    *elem = (&*elem - &mins[index]) / (&maxs[index] - &mins[index])
                }
            }
        }
        Scaler::ZScore => {
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
                    *elem = (&*elem - &means[index]) / &stds[index];
                }
            }
        }
    }
}
