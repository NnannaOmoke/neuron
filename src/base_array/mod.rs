#[cfg(test)]
mod base_matrix_tests;

mod base_dataset;

use crate::{
    dtype::{self, DType, DTypeType},
    *,
};
use ndarray::{ArrayView1, ArrayViewMut1, ShapeError};
use std::io::Read;
use thiserror::Error;

#[repr(C)]
#[derive(Clone)]
pub(crate) struct BaseMatrix {
    pub data: Array2<DType>,
}

impl BaseMatrix {
    //add some functionality to build from csv later
    pub(crate) fn transpose(self) -> Self {
        BaseMatrix {
            data: self.data.reversed_axes(),
        }
    }
    pub(crate) fn shape(&self) -> (usize, usize) {
        assert!(self.data.shape().len() == 2);
        (self.data.nrows(), self.data.ncols())
    }
    pub(crate) fn get_col(&self, cindex: usize) -> ArrayView1<'_, DType> {
        self.data.column(cindex)
    }
    pub(crate) fn get_mut_col(&mut self, cindex: usize) -> ArrayViewMut1<'_, DType> {
        self.data.column_mut(cindex)
    }
    pub(crate) fn get_row(&self, rindex: usize) -> ArrayView1<'_, DType> {
        self.data.row(rindex)
    }
    pub(crate) fn get_mut_row(&mut self, rindex: usize) -> ArrayViewMut1<'_, DType> {
        self.data.row_mut(rindex)
    }
    pub(crate) fn get(&self, rindex: usize, cindex: usize) -> &DType {
        self.data.get((rindex, cindex)).unwrap()
    }
    pub(crate) fn get_mut(&mut self, rindex: usize, cindex: usize) -> &mut DType {
        self.data.get_mut((rindex, cindex)).unwrap()
    }
    pub(crate) fn len(&self) -> usize {
        self.data.nrows()
    }
    pub(crate) fn push_col(&mut self, slice: &[DType]) {
        self.data
            .push_column(slice.into())
            .expect("Shape is not compatible")
    }
    pub(crate) fn push_row(&mut self, slice: &[DType]) {
        self.data
            .push_row(slice.into())
            .expect("Incompatible shapes!")
    }
    pub(crate) fn pop_col(&mut self) -> &[DType] {
        todo!()
    }
    pub(crate) fn cols(&self) -> ColumnIter<'_> {
        ColumnIter {
            inner: self.data.columns().into_iter(),
        }
    }
    pub(crate) fn rows(&self) -> RowIter<'_> {
        RowIter {
            inner: self.data.rows().into_iter(),
        }
    }
    pub(crate) fn try_from_csv<R: Read>(
        reader: csv::Reader<R>,
        prefer_precision: bool,
    ) -> Result<Self, Error> {
        // The nice thing about this code is that we don't have to check col sizes; `csv` does this
        // automatically.
        let mut col_types = Vec::new();
        let mut row = Vec::new();
        let mut records = reader.into_records();
        let first_record = if let Some(rr) = records.next() {
            rr?
        } else {
            return Err(Error::NoData);
        };
        for field in first_record.iter() {
            let field_data = DType::parse_from_str(field.trim(), prefer_precision);
            let field_data_type = field_data.data_type();
            col_types.push(field_data_type);
            row.push(field_data);
        }
        let mut arr = Array2::from_elem((0, row.len()), DType::None);
        arr.push_row(ArrayView1::from(&row))?;
        for record_res in records {
            let record = record_res?;
            for (i, field) in record.into_iter().enumerate() {
                let dtype = DType::parse_from_str(field.trim(), prefer_precision);
                // SAFETY: If the record we got had a different number of fields, the read method would
                // have failed based on the way the csv Reader works.
                let expected_data_type = row[i].data_type();
                let found_data_type = dtype.data_type();
                if found_data_type == expected_data_type {
                    row[i] = dtype;
                } else {
                    return Err(Error::MismatchedDTypes {
                        expected_data_type,
                        found_data_type,
                    });
                }
            }
            arr.push_row(ArrayView1::from(&row))?;
        }
        Ok(BaseMatrix { data: arr })
    }
}

impl AddAssign<&BaseMatrix> for BaseMatrix {
    fn add_assign(&mut self, rhs: &BaseMatrix) {
        assert_eq!(self.shape(), rhs.shape());
        self.data += &rhs.data
    }
}
impl Add<BaseMatrix> for BaseMatrix {
    type Output = BaseMatrix;
    fn add(self, mut rhs: BaseMatrix) -> Self::Output {
        rhs += &self;
        rhs
    }
}
impl SubAssign<BaseMatrix> for BaseMatrix {
    fn sub_assign(&mut self, rhs: BaseMatrix) {
        assert_eq!(self.shape(), rhs.shape());
        self.data -= &rhs.data
    }
}
impl SubAssign<&BaseMatrix> for BaseMatrix {
    fn sub_assign(&mut self, rhs: &BaseMatrix) {
        assert_eq!(self.shape(), rhs.shape());
        self.data -= &rhs.data
    }
}
//Subtraction is not implemented yet

// impl Index<usize> for BaseMatrix {
//     type Output = [DType];
//     fn index(&self, index: usize) -> &Self::Output {
//         &self[index] //this is row major, apparently, so this returns a row, I think
//     }
// }

// impl<T> std::ops::Mul<T> for BaseMatrix {
//     type Output = BaseMatrix;
//     fn mul(self, rhs: T) -> Self::Output {
//         self
//     }
// }

impl MulAssign<BaseMatrix> for BaseMatrix {
    fn mul_assign(&mut self, rhs: BaseMatrix) {
        self.data *= &rhs.data
    }
}

impl MulAssign<&BaseMatrix> for BaseMatrix {
    fn mul_assign(&mut self, rhs: &BaseMatrix) {
        self.data *= &rhs.data
    }
}
impl Mul<BaseMatrix> for BaseMatrix {
    type Output = BaseMatrix;
    fn mul(self, _rhs: BaseMatrix) -> Self::Output {
        //Uninmplemented because matrix multiplication is not commutative and i'll have to clone()
        self
    }
}

#[derive(Debug, Error)]
pub(crate) enum Error {
    #[error(transparent)]
    CsvError(#[from] csv::Error),
    #[error("columns of matrix were not all of same data type")]
    MismatchedDTypes {
        expected_data_type: DTypeType,
        found_data_type: DTypeType,
    },
    #[error("Matrix or source of data was empty")]
    NoData,
    #[error(transparent)]
    ShapeError(#[from] ShapeError),
}

pub(crate) struct ColumnIter<'a> {
    inner: LanesIter<'a, DType, Ix1>,
}

impl<'a> Iterator for ColumnIter<'a> {
    type Item = ArrayBase<ViewRepr<&'a DType>, Dim<[usize; 1]>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

pub(crate) struct RowIter<'a> {
    inner: LanesIter<'a, DType, Ix1>,
}

impl<'a> Iterator for RowIter<'a> {
    type Item = ArrayBase<ViewRepr<&'a DType>, Dim<[usize; 1]>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}
