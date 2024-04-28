use ndarray::{ArrayView1, ArrayViewMut2};

use super::*;
use crate::*;

pub fn forward_elimination(array: &mut ArrayViewMut2<f64>) {
    let len = array.shape()[0];
    for i in 0..len - 1 {
        for j in (i + 1)..len {
            let pivot = array[(j, i)] / array[(i, i)];
            if pivot.abs() < 1e-10f64 {
                eprintln!("Potential singular matrix encountered!");
                return;
            }
            for k in i + 1..=len {
                array[(j, k)] -= pivot * array[(i, k)];
            }
        }
    }
}

pub fn backward_substitution(array: &mut ArrayViewMut2<f64>) {
    let len = array.shape()[0];
    for i in (0..=(len - 1)).rev() {
        let mut sum = 0f64;
        for j in (i + 1)..len {
            sum += array[(i, j)] * array[(j, len)];
        }
        array[(i, len)] = (array[(i, len)] - sum) / array[(i, i)]
    }
}

pub fn solve_linear_systems(array: &mut ArrayViewMut2<f64>) {
    let len = array.shape()[0];
    if len == 0 {
        return;
    }
    forward_elimination(array);
    if array[(len - 1, len - 1)].abs() < 1e-10f64 {
        eprintln!("System has no unique solutions");
        return;
    }
    backward_substitution(array);
}

fn dot_d(left: ArrayView1<dtype::DType>, right: ArrayView1<dtype::DType>) -> f64 {
    assert!(left.len() == right.len());
    let mut res = dtype::DType::F64(0f64);
    zip(left.iter(), right.iter()).for_each(|(x, y)| {
        res += x * y;
    });
    res.to_f64().unwrap()
    //left.map(|x| x.to_f64().unwrap()).dot(&right.map(|x| x.to_f64().unwrap()))
}

pub fn dot(left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
    assert!(left.len() == right.len());
    left.dot(&right)
}

pub fn sigmoid(value: f64) -> f64{
    1f64/1f64 + (-value.exp())
}
