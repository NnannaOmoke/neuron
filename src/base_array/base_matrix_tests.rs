use super::{base_dataset::BaseDataset, *};
use crate::*;
use core::panic;
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
                DType::U32(16),
                DType::Object("male".to_string()),
                DType::U32(86)
            ],
            [
                DType::Object("james".to_string()),
                DType::U32(19),
                DType::Object("female".to_string()),
                DType::U32(86)
            ],
            [
                DType::Object("nissa".to_string()),
                DType::U32(6),
                DType::Object("female".to_string()),
                DType::U32(86)
            ],
            [
                DType::Object("karkarot".to_string()),
                DType::U32(893),
                DType::Object("saiyan".to_string()),
                DType::F32(86.4)
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
fn test_loader() {
    // let dataset = BaseDataset::from_csv(
    //     Path::new("src/base_array/test_data/diabetes.csv"),
    //     false,
    //     true,
    //     b',',
    // );
    // let mut dataset = dataset.unwrap();
    // let other_dataset = dataset.clone();
    // dataset.tail(None);
    // println!("{}", dataset.std(&"Glucose".to_string()));
    // println!("{}", dataset.total_memory_usage());
    // println!("{}", dataset.count(&"Glucose".to_string()));
    // //dataset.drop_col(&"Glucose".to_string());
    // dataset.head(None);
    // dataset.vstack(other_dataset);
    // dataset.head(None);
    let dataset = BaseDataset::from_matrix(
        BaseMatrix {
            data: arr2(&[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]).map(|v| DType::U32(*v)),
        },
        vec![
            "one".to_string(),
            "two".to_string(),
            "three".to_string(),
            "four".to_string(),
        ],
    );
    dbg!(dataset.sum(&"three".to_string()));
    panic!();
}
