use core::num;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use ndarray::{iter::LanesIter, Array2, ArrayBase, ArrayView, Axis, Dim, LinalgScalar, Ix1, Ix2, ViewRepr};

use crate::{dtype::DType};

#[repr(C)]
pub(crate) struct BaseMatrix{
    data: Array2<DType>
}

impl BaseMatrix{
    //@Sporadic Creator you're going to have to save me on this one :(
    // pub(crate) fn from_csv(sep: &str, fname: PathBuf) -> Self{ 
    // }
    pub(crate) fn transpose(self) -> Self{
        BaseMatrix{data: self.data.reversed_axes()}
    }
    pub(crate) fn shape(&self) -> (usize, usize){
        assert!(self.data.shape().len() == 2);
        (self.data.nrows(), self.data.ncols())
    }
    pub(crate) fn get_col(&self, cindex: usize) -> Option<&[DType]> {
        self.data.column(cindex).to_slice()
    }
    pub(crate) fn get_row(&self, rindex: usize) -> Option<&[DType]> {
        self.data.row(rindex).to_slice()
    }
    pub(crate) fn get(&self, rindex: usize, cindex: usize) -> DType{
        self.data.get((rindex, cindex)).unwrap().clone()
    }

    pub(crate) fn cols(&self) -> ColumnIter<'_> {
        ColumnIter {
            inner: self.data.columns().into_iter()
        }
    }

    pub(crate) fn rows(&self) -> RowIter<'_> {
        RowIter {
            inner: self.data.rows().into_iter()
        }
    }
}

//@ViableCompute, I want you to implement std::ops::traits for BaseMatrix [add, sub, mult(dot and element wise), div, index(use the get() function)]. When we're done with that we'll write a more userfriendly API that will
//be visible for our users, similar to pandas dataframe
impl std::ops::AddAssign<BaseMatrix> for BaseMatrix
{
    fn add_assign(&mut self, rhs: BaseMatrix) {
        assert_eq!(self.shape(), rhs.shape());
        self.data += &rhs.data
    }
}
impl std::ops::AddAssign<&BaseMatrix> for BaseMatrix
{
    fn add_assign(&mut self, rhs: &BaseMatrix) {
        assert_eq!(self.shape(), rhs.shape());
        self.data += &rhs.data
    }
}
impl std::ops::Add<BaseMatrix> for BaseMatrix
{
    type Output = BaseMatrix;
    fn add(self, mut rhs: BaseMatrix) -> Self::Output {
        rhs += &self;
        rhs
    }
}

impl std::ops::SubAssign<BaseMatrix> for BaseMatrix
{
    fn sub_assign(&mut self, rhs: BaseMatrix) {
        assert_eq!(self.shape(), rhs.shape());
        self.data -= &rhs.data
    }
}
impl std::ops::SubAssign<&BaseMatrix> for BaseMatrix
{
    fn sub_assign(&mut self, rhs: &BaseMatrix) {
        assert_eq!(self.shape(), rhs.shape());
        self.data -= &rhs.data
    }
}
//Subtraction is not implemented yet

impl std::ops::Index<usize> for BaseMatrix
{
    type Output = [DType];
    fn index(&self, index: usize) -> &Self::Output {
        self.get_row(index).unwrap()
    }
}

// impl<T> std::ops::Mul<T> for BaseMatrix {
//     type Output = BaseMatrix;
//     fn mul(self, rhs: T) -> Self::Output {
//         self
//     }
// }

impl std::ops::MulAssign<BaseMatrix> for BaseMatrix
{
    fn mul_assign(&mut self, rhs: BaseMatrix) {
        self.data *= &rhs.data
    }
}
impl std::ops::MulAssign<&BaseMatrix> for BaseMatrix
{
    fn mul_assign(&mut self, rhs: &BaseMatrix) {
        self.data *= &rhs.data
    }
}
impl std::ops::Mul<BaseMatrix> for BaseMatrix
{
    type Output = BaseMatrix;
    fn mul(self, rhs: BaseMatrix) -> Self::Output {
        //Uninmplemented because matrix multiplication is not commutative and i'll have to clone()
        self
    }
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
