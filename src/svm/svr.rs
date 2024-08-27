use crate::utils::model_selection::TrainTestSplitStrategy;
use crate::utils::model_selection::TrainTestSplitStrategyData;
use crate::utils::scaler::Scaler;
use crate::utils::scaler::ScalerState;
use ndarray::prelude::*;

use crate::svm::kernel_cache::CacheableSVM;
use crate::svm::kernel_cache::KernelCache;
use crate::svm::SVMKernel;
use crate::svm::{
    linear_kernel_1d, linear_kernel_2d, linear_kernel_mixed, polynomial_kernel_1d,
    polynomial_kernel_2d, polynomial_kernel_mixed, rbf_kernel_1d, rbf_kernel_2d, rbf_kernel_mixed,
    sigmoid_kernel_1d, sigmoid_kernel_2d, sigmoid_kernel_mixed,
};
#[derive(Clone)]
pub struct RawSVR {
    C: f64,
    eps: f64,
    support_vectors: Array2<f64>,
    support_labels: Array1<f64>,
    alphas: Array1<f64>,
    alpha_stars: Array1<f64>,
    bias: f64,
    kernel: SVMKernel,
}

impl CacheableSVM for RawSVR {
    fn kernel_op_1d(&self, row_one: ArrayView1<f64>, row_two: ArrayView1<f64>) -> f64 {
        match self.kernel {
            SVMKernel::RBF(gamma) => rbf_kernel_1d(row_one, row_two, gamma),
            SVMKernel::Linear => linear_kernel_1d(row_one, row_two),
            SVMKernel::Polynomial(degree, coef_) => {
                polynomial_kernel_1d(row_one, row_two, degree, coef_)
            }
            SVMKernel::Sigmoid(coef_, alpha) => sigmoid_kernel_1d(row_one, row_two, coef_, alpha),
        }
    }

    fn kernel_op_2d(&self, row_one: ArrayView2<f64>, row_two: ArrayView2<f64>) -> Array2<f64> {
        match self.kernel {
            SVMKernel::RBF(gamma) => rbf_kernel_2d(row_one, row_two, gamma),
            SVMKernel::Polynomial(degree, coef_) => {
                polynomial_kernel_2d(row_one, row_two, degree, coef_)
            }
            SVMKernel::Sigmoid(coef_, alpha) => sigmoid_kernel_2d(row_one, row_two, coef_, alpha),
            SVMKernel::Linear => linear_kernel_2d(row_one, row_two),
        }
    }

    fn kernel_op_mixed(&self, row_one: ArrayView1<f64>, row_two: ArrayView2<f64>) -> Array1<f64> {
        match self.kernel {
            SVMKernel::RBF(gamma) => rbf_kernel_mixed(row_one, row_two, gamma),
            SVMKernel::Polynomial(degree, coef_) => {
                polynomial_kernel_mixed(row_one, row_two, degree, coef_)
            }
            SVMKernel::Sigmoid(coef_, alpha) => {
                sigmoid_kernel_mixed(row_one, row_two, coef_, alpha)
            }
            SVMKernel::Linear => linear_kernel_mixed(row_one, row_two),
        }
    }

    fn kernel_op_helper(&self, i1: usize, i2: usize) -> f64 {
        self.kernel_op_1d(self.support_vectors.row(i1), self.support_vectors.row(i2))
    }
}

impl RawSVR {
    fn take_step(
        &mut self,
        i1: usize,
        i2: usize,
        error: &mut ArrayViewMut1<f64>,
        cache: &mut KernelCache,
    ) -> bool {
        if i1 == i2 {
            return false;
        }
        let alpha_one = self.alphas[i1];
        let alpha_one_star = self.alpha_stars[i2];
        let alpha_two = self.alphas[i1];
        let alpha_two_star = self.alpha_stars[i2];

        let error_one = error[i1];
        let error_two = error[i2];

        let k11 = cache.get([i1, i1].into(), self);
        let k12 = cache.get([i1, i2].into(), self);
        let k22 = cache.get([i2, i2].into(), self);
        let eta = 2f64 * k12 - k11 - k22;
        let gamma = (alpha_one - alpha_one_star) + (alpha_two - alpha_two_star);
        let [mut case1, mut case2, mut case3, mut case4, mut finished] = [0; 5];
        //cold storage for the current, but changeable alpha values
        let alpha_one_old = alpha_one;
        let alpha_two_old = alpha_two;
        let alpha_one_star_old = alpha_one_star;
        let alpha_two_star_old = alpha_two_star;

        let mut delta_phi = error_one - error_two;

        fn change(eps: f64, former: f64, new: f64) -> bool {
            if (new - former).abs() > eps {
                true
            } else {
                false
            }
        }

        while finished == 0 {
            if case1 == 0
                && (alpha_one > 0f64 || (alpha_one_star == 0f64 && delta_phi > 0f64))
                && (alpha_two > 0f64 || (alpha_two_star == 0f64 && delta_phi < 0f64))
            {
                let lower = 0f64.max(gamma - self.C);
                let upper = self.C.min(gamma);

                if lower < upper {
                    let mut alpha_two_new = alpha_two - delta_phi / eta;
                    alpha_two_new = alpha_two_new.min(upper);
                    alpha_two_new = alpha_two_new.max(lower);
                    let alpha_one_new = alpha_one - (alpha_two_new - alpha_two);
                    if change(self.eps, alpha_one, alpha_one_new)
                        && change(self.eps, alpha_two, alpha_two_new)
                    {
                        self.alphas[i1] = alpha_one_new;
                        self.alphas[i2] = alpha_two_new;
                    } else {
                        finished = 1;
                    }
                } else {
                    finished = 1;
                }
                case1 = 1;
            } else if case2 == 0
                && (alpha_one > 0f64 || (alpha_one_star == 0f64 && delta_phi > 2f64 * self.eps))
                && (alpha_two_star > 0f64 || (alpha_two == 0f64 && delta_phi < 2f64 * self.eps))
            {
                let lower = 0f64.max(gamma);
                let upper = self.C.min(self.C + gamma);

                if lower < upper {
                    let mut alpha_two_new = alpha_two_star + (delta_phi - 2f64 * self.eps) / eta;
                    alpha_two_new = alpha_two_new.min(upper);
                    alpha_two_new = alpha_two_new.max(lower);
                    let alpha_one_new = alpha_one + (alpha_two_new - alpha_two_star);
                    if change(self.eps, alpha_one, alpha_one_new)
                        && change(self.eps, alpha_two_star, alpha_two_new)
                    {
                        self.alphas[i1] = alpha_one_new;
                        self.alpha_stars[i2] = alpha_two_new;
                    } else {
                        finished = 1;
                    }
                } else {
                    finished = 1;
                }
                case2 = 1;
            } else if case3 == 0
                && (alpha_one_star > 0f64 || (alpha_one == 0f64 && delta_phi < 2f64 * self.eps))
                && (alpha_two > 0f64 || (alpha_two_star == 0f64 && delta_phi < 2f64 * self.eps))
            {
                let lower = 0f64.max(-gamma);
                let upper = self.C.min(-gamma + self.C);

                if lower < upper {
                    let mut alpha_two_new = alpha_two - (delta_phi - 2f64 * self.eps) / eta;
                    alpha_two_new = alpha_two_new.min(upper);
                    alpha_two_new = alpha_two_new.max(lower);
                    let alpha_one_new = alpha_one_star + (alpha_two_new - alpha_two);
                    if change(self.eps, alpha_one_star, alpha_one_new)
                        && change(self.eps, alpha_two, alpha_two_new)
                    {
                        self.alpha_stars[i1] = alpha_one_new;
                        self.alphas[i2] = alpha_two_new;
                    } else {
                        finished = 1;
                    }
                } else {
                    finished = 1;
                }
                case3 = 1;
            } else if case4 == 0
                && (alpha_one_star > 0f64 || (alpha_one == 0f64 && delta_phi < 0f64))
                && (alpha_two_star > 0f64 || (alpha_two == 0f64 && delta_phi > 0f64))
            {
                let lower = 0f64.max(-gamma - self.C);
                let upper = self.C.min(-gamma);

                if lower < upper {
                    let mut alpha_two_new = alpha_two_star + delta_phi / eta;
                    alpha_two_new = alpha_two_new.min(upper);
                    alpha_two_new = alpha_two_new.max(lower);
                    let alpha_one_new = alpha_one_star + (alpha_two_new - alpha_two_star);
                    if change(self.eps, alpha_one_star, alpha_one_new)
                        && change(self.eps, alpha_two_star, alpha_two_new)
                    {
                        self.alpha_stars[i1] = alpha_one_new;
                        self.alphas[i2] = alpha_two_new;
                    } else {
                        finished = 1;
                    }
                } else {
                    finished = 1;
                }
            }

            delta_phi = error_one
                - error_two
                - eta
                    * ((self.alphas[i1] - self.alpha_stars[i1])
                        - (alpha_one_old - alpha_one_star_old));
        }
        //updating the bias term
        let b1 = self.bias
            + error_one
            + self.support_labels[i1]
                * ((self.alphas[i1] - alpha_one_old) - (self.alpha_stars[i1] - alpha_one_star_old))
                * k11
            + self.support_labels[i2]
                * ((self.alphas[i2] - alpha_two_old) - (self.alpha_stars[i2] - alpha_two_star_old))
                * k12
            - self.eps;
        let b1_star = self.bias
            + error_one
            + self.support_labels[i1]
                * ((self.alphas[i1] - alpha_one_old) - (self.alpha_stars[i1] - alpha_one_star_old))
                * k11
            + self.support_labels[i2]
                * ((self.alphas[i2] - alpha_two_old) - (self.alpha_stars[i2] - alpha_two_star_old))
                * k12
            + self.eps;
        let b2 = self.bias
            + error_two
            + self.support_labels[i2]
                * ((self.alphas[i1] - alpha_one_old) - (self.alpha_stars[i1] - alpha_one_star_old))
                * k22
            + self.support_labels[i1]
                * ((self.alphas[i2] - alpha_two_old) - (self.alpha_stars[i2] - alpha_two_star_old))
                * k12
            - self.eps;
        let b2_star = self.bias
            + error_two
            + self.support_labels[i2]
                * ((self.alphas[i1] - alpha_one_old) - (self.alpha_stars[i1] - alpha_one_star_old))
                * k22
            + self.support_labels[i1]
                * ((self.alphas[i2] - alpha_two_old) - (self.alpha_stars[i2] - alpha_two_star_old))
                * k12
            + self.eps;
        let bias = if (0f64 < self.alphas[i1]) && (self.alphas[i1] < self.C) {
            b1
        } else if (0f64 < self.alpha_stars[i1]) && (self.alpha_stars[i1] < self.C) {
            b1_star
        } else if (0f64 < self.alphas[i2]) && (self.alpha_stars[i2] < self.C) {
            b2
        } else if (0f64 < self.alpha_stars[i2]) && (self.alpha_stars[i2] < self.C) {
            b2_star
        } else {
            //just because, honestly
            (b1 + b2) * 0.5
        };

        //no we've update bias, next step is to update the error cache
        let non_optimized = (0..self.support_vectors.len())
            .filter(|&index| index != i2 && index != i1)
            .collect::<Vec<usize>>();

        true
    }

    fn predict(&self, data: ArrayView2<f64>) -> Array1<f64> {
        let kernel = self.kernel_op_2d(self.support_vectors.view(), data);
        let alphas = self.alphas.to_owned() - self.alpha_stars.view();
        let weights = alphas * self.support_labels.view();
        let res = weights.dot(&kernel.view());
        res - self.bias
    }
}
