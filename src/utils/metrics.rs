#![allow(unused_variables)]
//temporary thing for now
use super::*;
use crate::{linear_models::LinearRegressorBuilder, *};
use ndarray::{ArrayView1, ArrayView2};
use num_traits::Pow;

pub fn mean_abs_error(target: ArrayView1<f64>, predicted: ArrayView1<f64>) -> Vec<f64> {
    assert!(target.len() == predicted.len());
    let len = target.len() as f64;
    let mut total = 0f64;
    zip(target, predicted).for_each(|(x, y)| total += (x - y).abs());
    vec![total / len]
}

pub fn mean_squared_error(target: ArrayView1<f64>, predicted: ArrayView1<f64>) -> Vec<f64> {
    assert!(target.len() == predicted.len());
    let len = target.len() as f64;
    let mut total = 0f64;
    zip(target, predicted).for_each(|(x, y)| total += (x - y).pow(2));
    vec![total / len]
}

pub fn root_mean_square_error(target: ArrayView1<f64>, predicted: ArrayView1<f64>) -> Vec<f64> {
    vec![mean_squared_error(target, predicted)[0].sqrt()]
}

pub fn accuracy(target: ArrayView1<u32>, predicted: ArrayView1<u32>) -> Vec<f64> {
    //probably need to create a heatmap for combination of variables
    assert!(target.len() == predicted.len());
    let correct = zip(target, predicted).filter(|(x, y)| x == y).count();
    vec![correct as f64 / target.len() as f64]
}

pub fn precision(target: ArrayView1<u32>, predicted: ArrayView1<u32>) -> f64 {
    todo!()
}

pub fn recall(target: ArrayView1<u32>, predicted: ArrayView1<u32>) -> f64 {
    todo!()
}

pub fn f1_score(target: ArrayView1<u32>, predicted: ArrayView1<u32>) -> f64 {
    todo!()
}
