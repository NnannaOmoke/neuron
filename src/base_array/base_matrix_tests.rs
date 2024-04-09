use crate::dtype::DType;

use super::{base_dataset::BaseDataset, BaseMatrix};
use std::{borrow::Cow, path::Path};

#[test]
pub fn test_base_matrix_try_from_csv() {
    let data = BaseDataset::from_csv(
        Path::new("src/base_array/test_data/test.csv"),
        false,
        true,
        b',',
    );
    let mut data = data.unwrap();
    data.head(None);
    println!("{:?}", data.columns());
    println!("{}", data.len());
    println!("{}", data.mode(&"age".to_string()));
    println!("{}", data.std(&"age".to_string()));
    data.clip("age".to_string(), 25f32.into(), 0f32.into());
    data.head(None);
    data.map("age".to_string(), |x|{*x = &*x * &DType::F32(2.0)});
    data.head(None);
    data.dtypes();
}
