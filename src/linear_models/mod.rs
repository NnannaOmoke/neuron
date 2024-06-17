use core::num;
use ndarray::{linalg, s, Array1, Array2, ArrayView1, ArrayViewMut1, ArrayViewMut2};
use ndarray_linalg::{solve::Inverse, InnerProduct, Scalar};

use crate::{
    base_array::{base_dataset::BaseDataset, BaseMatrix},
    dtype::DType,
    utils::{
        math::{dot, one_hot_encode_1d, softmax_1d, solve_linear_systems},
        model_selection::{self, TrainTestSplitStrategy, TrainTestSplitStrategyData},
        scaler::{Scaler, ScalerState},
    },
    *,
};

mod linear_regression;
mod logistic_regression;

pub(super) use linear_regression::LinearRegressorBuilder;
pub(super) use logistic_regression::LogisticRegressorBuilder;

#[derive(Clone, Copy)]
pub enum LinearRegularizer {
    None,
    Ridge(f64),
    Lasso(f64, usize),
    ElasticNet(f64, f64, usize),
}

pub fn _coordinate_descent(
    features: ArrayView2<f64>,
    target: ArrayView1<f64>,
    l1_regularizer: f64,
    l2_regularizer: Option<f64>,
    iters: usize,
) -> Array1<f64> {
    let ncols = features.shape()[1];
    let mut weights = Array1::ones(ncols);
    for _ in 0..iters {
        for (index, col) in features.columns().into_iter().enumerate() {
            let step = _compute_step_col(features.view(), target, weights.view(), index, col);
            let col_norm_factor = _compute_norm_term(col);
            weights[index] = match l2_regularizer {
                Some(var) => elastic_net_soft_threshold(step, l1_regularizer, var, col_norm_factor),
                None => lasso_soft_threshold(step, l1_regularizer, col_norm_factor),
            }
        }
    }
    weights
}

pub fn lasso_soft_threshold(rho: f64, lambda: f64, col_norm_factor: f64) -> f64 {
    if rho < -lambda {
        (rho + lambda) / col_norm_factor
    } else if rho > lambda {
        (rho - lambda) / col_norm_factor
    } else {
        0f64
    }
}

pub fn elastic_net_soft_threshold(rho: f64, l1: f64, l2: f64, col_norm_factor: f64) -> f64 {
    let gamma = if rho > 0f64 && (l1 * l2) < rho.abs() {
        rho - (l1 * l2)
    } else if rho < 0f64 && (l1 * l2) < rho.abs() {
        rho + (l1 * l2)
    } else {
        0f64
    };
    let gamma_interlude = gamma / (1f64 + l1 * (1f64 - l2));
    gamma_interlude / col_norm_factor
}

pub fn _compute_step_col(
    features: ArrayView2<f64>,
    target: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    index: usize,
    col: ArrayView1<f64>,
) -> f64 {
    let mut feature_clone = features.to_owned();
    feature_clone.remove_index(Axis(1), index);
    let mut weight_clone = weights.to_owned();
    weight_clone.remove_index(Axis(0), index);
    let prediction = feature_clone.dot(&weight_clone.view());
    let res = target.to_owned() - prediction;
    let step_col = col.dot(&res);
    step_col
}

pub fn _compute_norm_term(col: ArrayView1<f64>) -> f64 {
    col.dot(&col)
}

pub fn non_regularizing_fit(features: ArrayView2<f64>, target: ArrayView1<f64>) -> Array1<f64> {
    let feature_t = features.t();
    let left = feature_t.dot(&features);
    let right = feature_t.dot(&target);
    let left = left.inv().expect("Inversion Failed");
    let weights = left.dot(&right);
    weights
}

pub fn ridge_regularizing_fit(
    features: ArrayView2<f64>,
    target: ArrayView1<f64>,
    regularizer: f64,
) -> Array1<f64> {
    let feature_t = features.t();
    let eye = Array2::eye(features.ncols());
    let l_left = feature_t.dot(&features);
    let l_right = regularizer * eye;
    let left = l_left + l_right;
    let left_inv = left.inv().expect("Inversion Failed");
    let right = feature_t.dot(&target);
    left_inv.dot(&right)
}
