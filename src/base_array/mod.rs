use core::num;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use ndarray::{iter::LanesIter, ArrayBase, ArrayView, Axis, Dim, LinalgScalar, Ix1, Ix2, ViewRepr};

use crate::*;

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub enum DType {
    None,
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Object(String),
}

impl Add<&DType> for DType {
    type Output = DType;

    fn add(self, rhs: &DType) -> Self::Output {
        match self {
            // There has got to be a better way.
            DType::None => rhs.clone(),
            DType::U32(l) => match rhs {
                DType::None => DType::U32(l),
                DType::U32(r) => DType::U32(l + r),
                DType::U64(r) => DType::U64(l as u64 + r),
                DType::F32(r) => DType::F32(l as f32 + r),
                DType::F64(r) => DType::F64(l as f64 + r),
                DType::Object(_) => DType::U32(l),
            },
            DType::U64(l) => match rhs {
                DType::None => DType::U64(l),
                DType::U32(r) => DType::U64(l + *r as u64),
                DType::U64(r) => DType::U64(l + r),
                DType::F32(r) => DType::F32(l as f32 + r),
                DType::F64(r) => DType::F64(l as f64 + r),
                DType::Object(_) => DType::U64(l),
            },
            DType::F32(l) => match rhs {
                DType::None => DType::F32(l),
                DType::U32(r) => DType::F32(l + *r as f32),
                DType::U64(r) => DType::F32(l + *r as f32),
                DType::F32(r) => DType::F32(l + r),
                DType::F64(r) => DType::F64(l as f64 + r),
                DType::Object(_) => DType::F32(l),
            },
            DType::F64(l) => match rhs {
                DType::None => DType::F64(l),
                DType::U32(r) => DType::F64(l + *r as f64),
                DType::U64(r) => DType::F64(l + *r as f64),
                DType::F32(r) => DType::F64(l + *r as f64),
                DType::F64(r) => DType::F64(l + r),
                DType::Object(_) => DType::F64(l),
            },
            DType::Object(l) => match rhs {
                DType::Object(r) => DType::Object(l + r),
                _ => DType::Object(l),
            }

        }
    }
}

impl Add<&DType> for &DType {
    type Output = DType;

    fn add(self, rhs: &DType) -> Self::Output {
        match self {
            // There has got to be a better way.
            DType::None => rhs.clone(),
            DType::U32(l) => match rhs {
                DType::None => DType::U32(*l),
                DType::U32(r) => DType::U32(l + r),
                DType::U64(r) => DType::U64(*l as u64 + r),
                DType::F32(r) => DType::F32(*l as f32 + r),
                DType::F64(r) => DType::F64(*l as f64 + r),
                DType::Object(_) => DType::U32(*l),
            },
            DType::U64(l) => match rhs {
                DType::None => DType::U64(*l),
                DType::U32(r) => DType::U64(l + *r as u64),
                DType::U64(r) => DType::U64(l + r),
                DType::F32(r) => DType::F32(*l as f32 + r),
                DType::F64(r) => DType::F64(*l as f64 + r),
                DType::Object(_) => DType::U64(*l),
            },
            DType::F32(l) => match rhs {
                DType::None => DType::F32(*l),
                DType::U32(r) => DType::F32(l + *r as f32),
                DType::U64(r) => DType::F32(l + *r as f32),
                DType::F32(r) => DType::F32(l + r),
                DType::F64(r) => DType::F64(*l as f64 + r),
                DType::Object(_) => DType::F32(*l),
            },
            DType::F64(l) => match rhs {
                DType::None => DType::F64(*l),
                DType::U32(r) => DType::F64(l + *r as f64),
                DType::U64(r) => DType::F64(l + *r as f64),
                DType::F32(r) => DType::F64(l + *r as f64),
                DType::F64(r) => DType::F64(l + r),
                DType::Object(_) => DType::F64(*l),
            },
            DType::Object(l) => match rhs {
                DType::Object(r) => DType::Object(l.clone() + r),
                _ => DType::Object(l.clone()),
            }
        }
    }
}

impl Sub<&DType> for DType {
    type Output = DType;

    fn sub(self, rhs: &DType) -> Self::Output {
        match self {
            DType::None => match rhs {
                _ => DType::None,
            },
            DType::U32(l) => match rhs {
                DType::None => DType::U32(l),
                DType::U32(r) => DType::U32(l - r),
                DType::U64(r) => DType::U64(l as u64 - r),
                DType::F32(r) => DType::F32(l as f32 - r),
                DType::F64(r) => DType::F64(l as f64 - r),
                DType::Object(_) => DType::U32(l),
            },
            DType::U64(l) => match rhs {
                DType::None => DType::U64(l),
                DType::U32(r) => DType::U64(l - *r as u64),
                DType::U64(r) => DType::U64(l - r),
                DType::F32(r) => DType::F32(l as f32 - r),
                DType::F64(r) => DType::F64(l as f64 - r),
                DType::Object(_) => DType::U64(l),
            },
            DType::F32(l) => match rhs {
                DType::None => DType::F32(l),
                DType::U32(r) => DType::F32(l - *r as f32),
                DType::U64(r) => DType::F32(l - *r as f32),
                DType::F32(r) => DType::F32(l - r),
                DType::F64(r) => DType::F64(l as f64 - r),
                DType::Object(_) => DType::F32(l),
            },
            DType::F64(l) => match rhs {
                DType::None => DType::F64(l),
                DType::U32(r) => DType::F64(l - *r as f64),
                DType::U64(r) => DType::F64(l - *r as f64),
                DType::F32(r) => DType::F64(l - *r as f64),
                DType::F64(r) => DType::F64(l - r),
                DType::Object(_) => DType::F64(l),
            },
            DType::Object(l) => match rhs {
                _ => DType::Object(l),
            }

        }
    }
}

impl Sub<&DType> for &DType {
    type Output = DType;

    fn sub(self, rhs: &DType) -> Self::Output {
        match self {
            DType::None => match rhs {
                _ => DType::None,
            },
            DType::U32(l) => match rhs {
                DType::None => DType::U32(*l),
                DType::U32(r) => DType::U32(l - r),
                DType::U64(r) => DType::U64(*l as u64 - r),
                DType::F32(r) => DType::F32(*l as f32 - r),
                DType::F64(r) => DType::F64(*l as f64 - r),
                DType::Object(_) => DType::U32(*l),
            },
            DType::U64(l) => match rhs {
                DType::None => DType::U64(*l),
                DType::U32(r) => DType::U64(l - *r as u64),
                DType::U64(r) => DType::U64(l - r),
                DType::F32(r) => DType::F32(*l as f32 - r),
                DType::F64(r) => DType::F64(*l as f64 - r),
                DType::Object(_) => DType::U64(*l),
            },
            DType::F32(l) => match rhs {
                DType::None => DType::F32(*l),
                DType::U32(r) => DType::F32(l - *r as f32),
                DType::U64(r) => DType::F32(l - *r as f32),
                DType::F32(r) => DType::F32(l - r),
                DType::F64(r) => DType::F64(*l as f64 - r),
                DType::Object(_) => DType::F32(*l),
            },
            DType::F64(l) => match rhs {
                DType::None => DType::F64(*l),
                DType::U32(r) => DType::F64(l - *r as f64),
                DType::U64(r) => DType::F64(l - *r as f64),
                DType::F32(r) => DType::F64(l - *r as f64),
                DType::F64(r) => DType::F64(l - r),
                DType::Object(_) => DType::F64(*l),
            },
            DType::Object(l) => match rhs {
                _ => DType::Object(l.clone()),
            }

        }
    }
}

impl Mul<&DType> for DType {
    type Output = DType;

    fn mul(self, rhs: &DType) -> Self::Output {
        match self {
            // There has got to be a better way.
            DType::None => rhs.clone(),
            DType::U32(l) => match rhs {
                DType::None => DType::U32(l),
                DType::U32(r) => DType::U32(l * r),
                DType::U64(r) => DType::U64(l as u64 * r),
                DType::F32(r) => DType::F32(l as f32 * r),
                DType::F64(r) => DType::F64(l as f64 * r),
                DType::Object(_) => DType::U32(l),
            },
            DType::U64(l) => match rhs {
                DType::None => DType::U64(l),
                DType::U32(r) => DType::U64(l * *r as u64),
                DType::U64(r) => DType::U64(l * r),
                DType::F32(r) => DType::F32(l as f32 * r),
                DType::F64(r) => DType::F64(l as f64 * r),
                DType::Object(_) => DType::U64(l),
            },
            DType::F32(l) => match rhs {
                DType::None => DType::F32(l),
                DType::U32(r) => DType::F32(l * *r as f32),
                DType::U64(r) => DType::F32(l * *r as f32),
                DType::F32(r) => DType::F32(l * r),
                DType::F64(r) => DType::F64(l as f64 * r),
                DType::Object(_) => DType::F32(l),
            },
            DType::F64(l) => match rhs {
                DType::None => DType::F64(l),
                DType::U32(r) => DType::F64(l * *r as f64),
                DType::U64(r) => DType::F64(l * *r as f64),
                DType::F32(r) => DType::F64(l * *r as f64),
                DType::F64(r) => DType::F64(l * r),
                DType::Object(_) => DType::F64(l),
            },
            DType::Object(l) => match rhs {
                DType::Object(r) => DType::Object(l + r),
                _ => DType::Object(l),
            }

        }
    }
}

impl Mul<&DType> for &DType {
    type Output = DType;

    fn mul(self, rhs: &DType) -> Self::Output {
        match self {
            // There has got to be a better way.
            DType::None => rhs.clone(),
            DType::U32(l) => match rhs {
                DType::None => DType::U32(*l),
                DType::U32(r) => DType::U32(l * r),
                DType::U64(r) => DType::U64(*l as u64 * r),
                DType::F32(r) => DType::F32(*l as f32 * r),
                DType::F64(r) => DType::F64(*l as f64 * r),
                DType::Object(_) => DType::U32(*l),
            },
            DType::U64(l) => match rhs {
                DType::None => DType::U64(*l),
                DType::U32(r) => DType::U64(l * *r as u64),
                DType::U64(r) => DType::U64(l * r),
                DType::F32(r) => DType::F32(*l as f32 * r),
                DType::F64(r) => DType::F64(*l as f64 * r),
                DType::Object(_) => DType::U64(*l),
            },
            DType::F32(l) => match rhs {
                DType::None => DType::F32(*l),
                DType::U32(r) => DType::F32(l * *r as f32),
                DType::U64(r) => DType::F32(l * *r as f32),
                DType::F32(r) => DType::F32(l * r),
                DType::F64(r) => DType::F64(*l as f64 * r),
                DType::Object(_) => DType::F32(*l),
            },
            DType::F64(l) => match rhs {
                DType::None => DType::F64(*l),
                DType::U32(r) => DType::F64(l * *r as f64),
                DType::U64(r) => DType::F64(l * *r as f64),
                DType::F32(r) => DType::F64(l * *r as f64),
                DType::F64(r) => DType::F64(l * r),
                DType::Object(_) => DType::F64(*l),
            },
            DType::Object(l) => match rhs {
                DType::Object(r) => DType::Object(l.clone() + r),
                _ => DType::Object(l.clone()),
            }

        }
    }
}

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
