use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2};
use ndarray_linalg::SVDInplace;

use crate::base_array::BaseDataset;

pub struct PCA {
    n_features: usize,
    rsv_ut: Array2<f64>,
}

impl PCA {
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            rsv_ut: Array2::default((0, 0)),
        }
    }

    pub fn fit(&mut self, dataset: &BaseDataset) -> Array2<f64> {
        let mut features = dataset.into_f64_array();
        //center the data
        features.columns_mut().into_iter().for_each(|mut col| {
            col -= col.mean().unwrap();
        });
        let rsv_t = Self::svd(&mut features.view_mut());
        let rsv_ut = Array2::from_shape_fn((features.nrows(), self.n_features), |(col, row)| {
            rsv_t[(row, col)]
        });
        self.rsv_ut = rsv_ut;
        features
    }

    pub fn transform(&self, input: Array2<f64>) -> Array2<f64> {
        self.rsv_ut.dot(&input)
    }

    pub fn fit_transform(&mut self, dataset: BaseDataset) -> Array2<f64> {
        let f = self.fit(&dataset);
        self.transform(f)
    }
    pub fn svd(input: &mut ArrayViewMut2<f64>) -> Array2<f64> {
        let (_, _, rsvt) = input.svd_inplace(false, true).unwrap();
        rsvt.unwrap()
    }
}
