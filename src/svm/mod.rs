//svm kernels
use ndarray::{ArrayView1, Array1, ArrayView2, Array2, array};
use ndarray_linalg::{norm::Norm, opnorm::OperationNorm};
use std::ops::Neg;

pub enum SVMKernel{
    RBF(f64), 
    Polynomial(u32, f64),
    Linear,
    Sigmoid(f64, f64)
}

pub fn linear_kernel_1d(input_one: ArrayView1<f64>, input_two: ArrayView1<f64>) -> f64{
    input_one.to_owned().dot(&input_two.t())
}

pub fn linear_kernel_2d(input_one: ArrayView2<f64>, input_two: ArrayView2<f64>) -> Array2<f64>{
    input_one.to_owned().dot(&input_two.t())
}

pub fn rbf_kernel_1d(input_one: ArrayView1<f64>, input_two: ArrayView1<f64>, gamma: f64) -> f64{
    let distance = (input_one.to_owned() - input_two).map(|x| x.powi(2)).sum();
    (gamma.neg() * distance).exp()
}

pub fn polynomial_kernel_1d(input_one: ArrayView1<f64>, input_two: ArrayView1<f64>, degree: u32, coef_: f64) -> f64{
    (input_one.to_owned().dot(&input_two) + coef_).powi(degree as i32)
}

pub fn sigmoid_kernel_1d(input_one: ArrayView1<f64>, input_two: ArrayView1<f64>, coef_: f64, alpha: f64) -> f64{
    (alpha * input_one.to_owned().dot(&input_two) + coef_).tanh()
}



#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test_rbf_kernel(){
        let first = array![1.0, 4.0, 8.0, 9.0];
        let second = array![4.0, 5.0, 6.0, 7.0];
        assert_eq!(rbf_kernel_1d(first.view(), second.view(), 1.0), 1.522997974471263e-08)
    }
}