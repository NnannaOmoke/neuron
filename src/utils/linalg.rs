use ndarray::ArrayViewMut2;

use super::*;
use crate::*;

pub fn forward_elimination(array: &mut ArrayViewMut2<f64>) {
    let len = array.len();
    for i in 0..len - 1 {
        for j in i + 1..len {
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
    let len = array.len();
    for i in (0..=len - 1).rev() {
        let mut sum = 0f64;
        for j in i + 1..len {
            sum += array[(i, j)] * array[(j, len)];
        }
        array[(i, len)] = (array[(i, len)] - sum) / array[(i, i)]
    }
}

pub fn solve_linear_systems(array: &mut ArrayViewMut2<f64>) {
    let len = array.len();
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
