use super::{base_dataset::BaseDataset, BaseMatrix};
use std::{borrow::Cow, path::Path};

#[test]
pub fn test_base_matrix_try_from_csv() {
    let data = BaseDataset::from_csv(
        Path::new("src/base_array/test_data/test.csv"),
        false,
        false,
        b',',
    );
    let data = data.unwrap();
    data.head(None);
    println!("{:?}", data.columns());
    println!("{}", data.len());
}
