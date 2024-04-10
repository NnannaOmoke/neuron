use super::{*, base_dataset::BaseDataset};
use crate::*;
use csv::Reader;
use ndarray::{arr2, Array2};
use std::{borrow::Cow, fs::File, path::Path};

#[test]
pub fn test_base_matrix_try_from_csv() {
    let data = BaseMatrix::try_from_csv(
        Reader::from_path("src/base_array/test_data/test.csv").unwrap(),
        false,
    )
    .unwrap();
    // TODO: Change if DType is changed to prioritze integer parsing.
    assert_eq!(
        data.data,
        arr2(&[
            [
                DType::Object("adamu".to_string()),
                DType::F32(16.0),
                DType::Object("male".to_string()),
                DType::F32(86.4)
            ],
            [
                DType::Object("james".to_string()),
                DType::F32(19.0),
                DType::Object("female".to_string()),
                DType::F32(74.0)
            ],
            [
                DType::Object("nissa".to_string()),
                DType::F32(6.0),
                DType::Object("female".to_string()),
                DType::F32(15.0)
            ],
            [
                DType::Object("karkarot".to_string()),
                DType::F32(893.0),
                DType::Object("saiyan".to_string()),
                DType::F32(100.0)
            ]
        ])
    );
}

#[test]
fn test_transpose() {
    let inner_data = Array2::from_shape_fn((2, 2), |(i, j)| DType::U32(i as u32 + j as u32 * 2));
    assert_eq!(
        inner_data,
        arr2(&[
            [DType::U32(0), DType::U32(2)],
            [DType::U32(1), DType::U32(3)],
        ])
    );
    let mut data = BaseMatrix { data: inner_data };
    data = data.transpose();
    assert_eq!(
        data.data,
        arr2(&[
            [DType::U32(0), DType::U32(1)],
            [DType::U32(2), DType::U32(3)],
        ])
    );
}

#[test]
fn test_loader(){
    let dataset = BaseDataset::from_csv(Path::new("src/base_array/test_data/diabetes.csv"), false, false, b',');
    let mut dataset = dataset.unwrap();
    dataset.head(None);
    dataset.dtypes();
}