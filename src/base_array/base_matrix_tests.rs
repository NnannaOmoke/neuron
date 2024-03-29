use super::{base_dataset::BaseDataset, BaseMatrix};
use std::{borrow::Cow, path::Path};

static CSV1: &'static str = include_str!("test_data/test.csv");

#[test]
pub fn test_base_matrix_try_from_csv() {
    let csv_reader = csv::ReaderBuilder::new().from_reader(CSV1.as_bytes());
    let data = BaseDataset::try_from_csv_reader(
        csv_reader,
        false,
        false,
        Cow::Borrowed(&[
            Cow::Borrowed("name"),
            Cow::Borrowed("age"),
            Cow::Borrowed("gender"),
            Cow::Borrowed("weight"),
        ]),
    );
    let data = data.unwrap();
    data.head(Some(2));
}
