use crate::svm::svc::RawSVC;
use crate::utils::math::outer_product;
use ndarray::{array, linalg::Dot, Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::{norm::Norm, opnorm::OperationNorm};
use std::ops::Neg;
pub mod svc;

#[derive(Clone, Default, Debug)]
//the SVM kernels to use
pub enum SVMKernel {
    RBF(f64),
    Polynomial(u32, f64),
    #[default]
    Linear,
    Sigmoid(f64, f64),
}

pub fn linear_kernel_1d(input_one: ArrayView1<f64>, input_two: ArrayView1<f64>) -> f64 {
    input_one.to_owned().dot(&input_two.t())
}

pub fn linear_kernel_2d(input_one: ArrayView2<f64>, input_two: ArrayView2<f64>) -> Array2<f64> {
    input_one.to_owned().dot(&input_two.t())
}

pub fn linear_kernel_mixed(input_one: ArrayView1<f64>, input_two: ArrayView2<f64>) -> Array1<f64> {
    input_one.to_owned().dot(&input_two.t())
}

pub fn rbf_kernel_1d(input_one: ArrayView1<f64>, input_two: ArrayView1<f64>, gamma: f64) -> f64 {
    let distance = (input_one.to_owned() - input_two).map(|x| x.powi(2)).sum();
    (gamma.neg() * distance).exp()
}

pub fn rbf_kernel_2d(
    input_one: ArrayView2<f64>,
    input_two: ArrayView2<f64>,
    gamma: f64,
) -> Array2<f64> {
    let mut allocated = Array2::from_elem((input_one.nrows(), input_two.nrows()), 0f64);
    for (index_one, row) in input_one.rows().into_iter().enumerate() {
        for (index_two, row_two) in input_two.rows().into_iter().enumerate() {
            let mut curr_distance = row.to_owned() - row_two;
            curr_distance.iter_mut().for_each(|x| *x = x.powi(2));
            let summation = curr_distance.sum();
            allocated[(index_one, index_two)] = (summation * gamma.neg()).exp();
        }
    }
    allocated
}

pub fn rbf_kernel_mixed(
    input_one: ArrayView1<f64>,
    input_two: ArrayView2<f64>,
    gamma: f64,
) -> Array1<f64> {
    let mut allocated = Array1::from_elem(input_two.nrows(), 0f64);
    for (index, row) in input_two.rows().into_iter().enumerate() {
        let mut distance = row.to_owned() - input_one;
        distance.iter_mut().for_each(|x| *x = x.powi(2));
        let summation = distance.sum();
        allocated[index] = (summation * gamma.neg()).exp();
    }
    allocated
}

pub fn polynomial_kernel_1d(
    input_one: ArrayView1<f64>,
    input_two: ArrayView1<f64>,
    degree: u32,
    coef_: f64,
) -> f64 {
    (input_one.to_owned().dot(&input_two) + coef_).powi(degree as i32)
}

pub fn polynomial_kernel_2d(
    input_one: ArrayView2<f64>,
    input_two: ArrayView2<f64>,
    degree: u32,
    coef_: f64,
) -> Array2<f64> {
    (input_one.to_owned().dot(&input_two) + coef_)
        .map(|x| x.powi(degree as i32))
        .to_owned()
}

pub fn polynomial_kernel_mixed(
    input_one: ArrayView1<f64>,
    input_two: ArrayView2<f64>,
    degree: u32,
    coef_: f64,
) -> Array1<f64> {
    (input_one.to_owned().dot(&input_two) + coef_)
        .map(|x| x.powi(degree as i32))
        .to_owned()
}

pub fn sigmoid_kernel_1d(
    input_one: ArrayView1<f64>,
    input_two: ArrayView1<f64>,
    coef_: f64,
    alpha: f64,
) -> f64 {
    (alpha * input_one.to_owned().dot(&input_two) + coef_).tanh()
}

pub fn sigmoid_kernel_2d(
    input_one: ArrayView2<f64>,
    input_two: ArrayView2<f64>,
    coef_: f64,
    alpha: f64,
) -> Array2<f64> {
    (alpha * input_one.to_owned().dot(&input_two) + coef_)
        .map(|x| x.tanh())
        .to_owned()
}

pub fn sigmoid_kernel_mixed(
    input_one: ArrayView1<f64>,
    input_two: ArrayView2<f64>,
    coef_: f64,
    alpha: f64,
) -> Array1<f64> {
    (alpha * input_one.to_owned().dot(&input_two) + coef_)
        .map(|x| x.tanh())
        .to_owned()
}

#[cfg(test)]
mod tests {
    use crate::utils::math::into_column_matrix;

    use super::*;

    #[test]
    fn test_rbf_kernel() {
        let one_shot = Array1::linspace(10.0, 14.0, 4);
        let support_vectors = array![
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0]
        ];
        let result = rbf_kernel_mixed(one_shot.view(), support_vectors.view(), 0.001);
        let result_two = rbf_kernel_2d(
            into_column_matrix(one_shot.view()).view(),
            support_vectors.view(),
            0.001,
        );
        dbg!(result);
        dbg!(result_two);
    }
}
