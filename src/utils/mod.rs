pub mod math;
pub mod metrics;
pub mod model_selection;
pub mod scaler;
pub mod text;

use crate::base_array::BaseDataset;
use crate::base_array::BaseMatrix;
use crate::dtype;
use crate::dtype::DType;
use crate::dtype::DTypeType;
use num_traits::Zero;
use std::collections::HashSet;
use std::iter::zip;

use neuron_macros::dtype;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2};

//performs one hot encoding for a base_dataset
pub fn one_hot_encode(dataset: BaseDataset, exclude: &[&str]) -> BaseDataset {
    //have this return the value_counts[1..]
    //then use said value counts to name the cols
    //we don't recompute because it can be returned in an arbitrary manner later
    fn raw_encode(
        array: &mut ArrayViewMut2<DType>,
        dataset: &BaseDataset,
        col: &str,
        start: &mut usize,
    ) -> Vec<String> {
        let index = dataset._get_string_index(col);
        let value_counts = dataset.value_counts(col).into_keys().collect::<Vec<_>>();
        let length = value_counts.len() - 1;
        let mut zeros = Array2::zeros((dataset.nrows(), length));
        dataset
            .get_col(index)
            .into_iter()
            .enumerate()
            .for_each(|(index, value)| {
                if value.clone() == value_counts[0] {
                } else {
                    let pos = value_counts[1..]
                        .iter()
                        .position(|current| value.clone() == *current)
                        .unwrap();
                    zeros[(index, pos)] = dtype!(1);
                }
            });
        //write the zeros to the array
        zip(0..length, *start..*start + length).for_each(|(zeros_index, array_index)| {
            array
                .column_mut(array_index)
                .assign(&zeros.column(zeros_index));
        });
        *start += length;
        value_counts[1..]
            .to_owned()
            .iter()
            .map(|x| x.to_string())
            .collect()
    }

    let allcols = dataset
        .columns()
        .iter()
        .filter(
            |colname| match dataset.get_first_nononetype(colname).data_type() {
                DTypeType::Object => true,
                _ => false,
            },
        )
        .map(|colname| colname.clone())
        .collect::<HashSet<_>>();
    let excluded = exclude
        .iter()
        .map(|s| s.to_string())
        .collect::<HashSet<_>>();
    let items = allcols.difference(&excluded).collect::<Vec<_>>();
    let extracols = items
        .iter()
        .map(|s| dataset.nunique(&s))
        .fold(0, |accum, num| accum + num - 1)
        - items.len();
    let mut data = Array2::from_elem(
        (dataset.nrows(), dataset.ncols() + extracols),
        DType::zero(),
    );
    let mut fixed = 0;
    let mut current = 0;
    let current_ref = &mut current;
    let mut colnames = vec![];
    for col in dataset.columns() {
        if col == items[fixed] {
            //handle it the other way
            let mut to_append = raw_encode(&mut data.view_mut(), &dataset, col, current_ref);
            to_append.push(col.to_string());
            colnames.push(to_append);
            fixed += 1;
        }
        //otherwise, memcopy
        else {
            data.column_mut(*current_ref)
                .assign(&dataset.get_col(dataset._get_string_index(col)));
            *current_ref += 1;
            colnames.push(vec![col.into()]);
        }
    }
    //flat map colnames
    colnames.iter_mut().for_each(|vector| {
        if vector.len() == 1 {
        } else {
            let last = vector.pop().unwrap();
            vector
                .iter_mut()
                .for_each(|elem| *elem = format!("{}_{}", last, elem));
        }
    });

    let cols = colnames
        .iter()
        .flatten()
        .map(|s| s.to_owned())
        .collect::<Vec<_>>();
    BaseDataset {
        data: BaseMatrix { data },
        column_names: cols,
    }
}
