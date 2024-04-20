use super::*;
use crate::{linear_models::LinearRegressorBuilder, *};

fn abs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}
pub fn mean_abs_error(target: &[f64], predicted: &[f64]) -> f64 {
    assert!(target.len() == predicted.len());
    let len = target.len() as f64;
    let mut total = 0f64;
    zip(target.iter(), predicted.iter()).for_each(|(x, y)| total += abs(x - y));
    total / len
}
