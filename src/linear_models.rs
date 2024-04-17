use core::num;

use float_derive::utils::eq;
use ndarray::{s, Array1, Array2, ArrayView1};
use num_traits::ToPrimitive;
use rand::{random, thread_rng, Rng};

use crate::{
    base_array::{base_dataset::BaseDataset, BaseMatrix},
    dtype::DType,
    *, utils::linalg::solve_linear_systems,
};

pub(crate) struct LinearRegressorBuilder {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    num_iters: usize,
}

impl LinearRegressorBuilder {
    fn new() -> Self {
        Self {
            weights: vec![],
            bias: 0f64,
            learning_rate: 0.00875f64,
            num_iters: 30000,
        }
    }

    fn all_colunms_dot(
        dataset: &BaseDataset,
        target_col: usize,
        dataset_index: usize,
        weights: &Vec<f64>,
    ) -> f64 {
        let num_cols = dataset.shape().1;
        let mut result = 0.0;
        let mut wt_index = 0usize;
        for i in 0..num_cols {
            if i == target_col {
                continue;
            }
            result += dataset.get(dataset_index, i).to_f64().unwrap() * weights[wt_index];
            wt_index += 1;
        }
        result
    }
    fn fit(&mut self, dataset: &BaseDataset, target_col: usize) {
        //if theres logic errors its probable in colunm_dot or because i mixed row and colunm somewhere
        let num_weights = dataset.shape().1 - 1;
        let n = dataset.shape().0;
        let n_inv = 1.0 / (n as f64);

        self.weights.resize(num_weights, 0.0);
        self.bias = 0.0;
        let mut dws: Vec<f64> = (0..num_weights).map(|_| random::<f64>()).collect();
        let mut db = 0.0;
        for _current_iter in 0..self.num_iters {
            for dataset_index in 0..n {
                //I couldn't figure out how to use ndarray's dot while skipping a colunm
                let error = self.bias
                    + Self::all_colunms_dot(dataset, target_col, dataset_index, &self.weights)
                    - dataset.get(dataset_index, target_col).to_f64().unwrap();
                let mut col_index = 0usize;
                for dw in &mut dws {
                    if col_index == target_col {
                        continue;
                    }
                    *dw += dataset.get(dataset_index, col_index).to_f64().unwrap() * error * n_inv;
                    col_index += 1;
                }
                db += error;
            }
            db *= n_inv;

            self.bias -= self.learning_rate * db;
            for i in 0..num_weights {
                self.weights[i] -= dws[i] * self.learning_rate;
            }
        }
    }

    pub fn weights(&self) -> &Vec<f64> {
        &self.weights
    }

    fn normalize(dataset: &mut BaseDataset, standardizer: Standardizer, targetcol: usize) {
        match standardizer {
            Standardizer::MinMax => {
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
            Standardizer::ZScore => {
                unimplemented!()
            }
        }
    }

    fn nfit(&mut self, dataset: &BaseDataset, target_col: &str) {
        assert!(self.num_iters != 0);
        let col_index = dataset._get_string_index(target_col);
        let (nsamples, nfeatures) = dataset.shape();
        println!("{:?}", dataset.shape());
        let mut rng = thread_rng();
        //turn everything
        let x = dataset.into_f64_array_without_target(col_index);
        let target = dataset
            .get_col(col_index)
            .map(|x| x.to_f64().unwrap())
            .into_owned();
        let mut weights = Array1::from_vec(
            (1..nfeatures)
                .map(|_| rng.gen::<f64>())
                .collect::<Vec<f64>>(),
        );
        let mut preds = Array1::from_elem(nsamples, 0f64);
        let mut dw = Array1::from_elem(nsamples, 0f64);
        let mut db = 0f64;
        for _ in 0..self.num_iters {
            preds = x.dot(&weights) + self.bias;
            dw = (1f64 / nsamples as f64) * (x.t().dot(&(&preds - &target)));
            db = (1f64 / nsamples as f64) * (&preds - &target).sum();
            weights = weights - &(self.learning_rate * dw);
            self.bias = self.bias - self.learning_rate * db;
        }
        self.weights = weights.to_vec();
    }

    fn predict(&self, data: &Array2<f64>) -> Vec<f64> {
        let mut predictions = (0..data.shape()[1]).map(|_| 0f64).collect::<Vec<f64>>();

        predictions
    }

    fn fitv2(&mut self, dataset: &BaseDataset, target: &str) {
        let target_col_index = dataset._get_string_index(target);
        let target = dataset.get_col(target_col_index);
        let (_, ncols) = dataset.shape();
        let nweights = ncols - 1; //cause we're taking in the full dataset; makes sense
        let mut eqns = Array2::from_elem((0, ncols), 0f64); //we don't have info about the shape of this array
        let mut first_ = Array1::from_elem(ncols, 0f64);
        (0..ncols)
            .filter(|index| *index != target_col_index)
            .for_each(|index| {
                first_[index] = dataset.get_col(index).sum().to_f64().unwrap();
            });
        //pushes the target col to the last in eqns; nice
        first_[ncols - 1] = target.sum().to_f64().unwrap();
        eqns.push_row(first_.view()).expect("First eqn couldn't fit in");
        let nsums = ((ncols - 1) * (ncols)) / 2;
        let mut sums: Vec<f64> = Vec::from_iter((0..nsums).map(|_| 0f64));
        for elem in 0 .. nsums {
            let mut first_col = 0;
            let mut group_ind: isize = elem as isize;
            loop {
                group_ind -= nweights as isize - first_col as isize;
                if group_ind < 0 {
                    break;
                }
                first_col += 1;
            }
            let mut second_col =
                (elem as isize - ((nweights as isize * 2 - first_col as isize + 1) * first_col as isize / 2)) + first_col as isize;
            first_col = if first_col < target_col_index {
                first_col
            } else {
                first_col + 1
            };
            second_col = if second_col < target_col_index as isize{
                second_col
            } else {
                second_col + 1
            };
            sums[elem] = dataset
                .get_col(first_col)
                .map(|x| x.to_f64().unwrap())
                .dot(&dataset.get_col(second_col as usize).map(|x| x.to_f64().unwrap()));
        }
        for elem in 0 .. nweights {
            let mut current = Vec::new();
            //bias
            current.push(eqns[(0, elem + 1)]);
            for elem_two in 0 .. nweights {
                println!("{}", elem_two);
                current.push(sums[Self::_sum_index(elem, elem_two, nweights)]);
            }
            let non_target_index = if elem < target_col_index {
                elem
            } else {
                elem + 1
            };
            let dot = dataset
                .get_col(non_target_index)
                .map(|x| x.to_f64().unwrap())
                .dot(
                    &dataset
                        .get_col(target_col_index)
                        .map(|x| x.to_f64().unwrap()),
                );
            current.push(dot);
            println!("Current length: {}", current.len());
            eqns.push_row(Array1::from_vec(current).view()).expect("Shape error");
        }

        solve_linear_systems(&mut eqns.view_mut());
        self.bias = eqns[(0, nweights)];
        for elem in 1 ..= nweights{
            self.weights[elem] = eqns[(elem, nweights)];
        }
    }

    fn _sum_index(eqn: usize, param: usize, nweights: usize) -> usize {
        let first = usize::min(eqn, param);
        let second = usize::max(eqn, param);
        let mut s_index = 0;
        s_index = (nweights * 2 - first + 1) * first / 2;
        s_index += second - first;
        s_index
    }
}

enum Standardizer {
    MinMax,
    ZScore,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    #[test]
    fn test_convergence() {
        let mut dataset = base_array::base_dataset::BaseDataset::from_csv(
            Path::new("src/base_array/test_data/boston.csv"),
            true,
            true,
            b',',
        )
        .unwrap();
        dataset.drop_na(None, true);
        let mut learner = LinearRegressorBuilder::new();
        LinearRegressorBuilder::normalize(&mut dataset, Standardizer::MinMax, 13);
        learner.fitv2(&dataset, "MEDV");
        println!("MAE: {}", 8)
    }
}
