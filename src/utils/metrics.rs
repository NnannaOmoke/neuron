#![allow(unused_variables)]
//temporary thing for now
use num_traits::Pow;

use super::*;
use crate::{linear_models::LinearRegressorBuilder, *};


pub fn mean_abs_error(target: &[f64], predicted: &[f64]) -> f64 {
    assert!(target.len() == predicted.len());
    let len = target.len() as f64;
    let mut total = 0f64;
    zip(target, predicted).for_each(|(x, y)| total += (x - y).abs());
    total / len
}

pub fn mean_squared_error(target: &[f64], predicted: &[f64]) -> f64 {
    assert!(target.len() == predicted.len());
    let len = target.len() as f64;
    let mut total = 0f64;
    zip(target, predicted).for_each(|(x, y)| total += (x - y).pow(2));
    total / len
}

pub fn root_mean_square_error(target: &[f64], predicted: &[f64]) -> f64 {
    mean_squared_error(target, predicted).sqrt()
}

pub fn accuracy(target: &[i32], predicted: &[i32]) -> f64 {
    //probably need to create a heatmap for combination of variables
    assert!(target.len() == predicted.len());
    todo!()
}

pub fn precision(target: &[i32], predicted: &[i32]) -> f64 {
    todo!()
}

pub fn recall(target: &[i32], predicted: &[i32]) -> f64 {
    todo!()
}

pub fn f1_score(target: &[i32], predicted: &[i32]) -> f64 {
    todo!()
}
