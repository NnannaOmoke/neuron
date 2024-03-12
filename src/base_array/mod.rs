use core::num;
use std::ops::{AddAssign, MulAssign, Sub, SubAssign};

use ndarray::{ArrayBase, ArrayView, LinalgScalar, ViewRepr};

use crate::*;
//we might need to define types for the dataset. each column can only have one type associated with it
pub(crate) type float = f32;
pub(crate) type int = i32;
pub(crate) type unsigned_int = u32;
pub(crate) type double = f64;
pub(crate) type long = i64;
pub(crate) type long_long = i128;
//we'll have to so something about strings
//if we have a numerical value that we can't parse, assume that the whole column is a string col
pub(crate) type object = String;

//These are the base types that each column must possess
//to use them, we can take the types we have, alias them, and impl `BaseType` for them

impl BaseType<float> for float{}
impl BaseType<int> for int{}
impl BaseType<unsigned_int> for unsigned_int{}
impl BaseType<double> for double{}
impl BaseType<long> for long{}
impl BaseType<long_long> for long_long{}


pub(crate) trait BaseType<T>
where T: Sized + LinalgScalar + AddAssign<T> + SubAssign<T>{}

#[repr(C)]
pub(crate) struct BaseMatrix<A: BaseType<A> + LinalgScalar + AddAssign<A> + SubAssign<A>>{
    data: Array2<A>
}

impl<A: BaseType<A> + LinalgScalar + AddAssign<A> + SubAssign<A>> BaseMatrix<A>{
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
    pub(crate) fn get_col(&self, cindex: usize) -> &[A]{
        self.data.column(cindex).to_slice().unwrap()
    }
    pub(crate) fn get_row(&self, rindex: usize) -> &[A]{
        self.data.row(rindex).to_slice().unwrap()
    }
    pub(crate) fn get(&self, rindex: usize, cindex: usize) -> A{
        *self.data.get((rindex, cindex)).unwrap()
    }
}
//here are the tasks: i.e. @sporadic_creator, please implement two iterators, one that will yield the cols, and the other the rows
//i.e. a row iterator that yields a tuple containing the data in a row and a column iterator that yields the data in a column

//@ViableCompute, I want you to implement std::ops::traits for BaseMatrix [add, sub, mult(dot and element wise), div, index(use the get() function)]. When we're done with that we'll write a more userfriendly API that will
//be visible for our users, similar to pandas dataframe
impl<A: BaseType<A> + AddAssign<A> + LinalgScalar + SubAssign<A>> std::ops::AddAssign<BaseMatrix<A>> for BaseMatrix<A>
{
    fn add_assign(&mut self, rhs: BaseMatrix<A>) {
        assert_eq!(self.shape(), rhs.shape());
        self.data += &rhs.data
    }
}
impl<A: BaseType<A> + AddAssign<A> + LinalgScalar + SubAssign<A>> std::ops::AddAssign<&BaseMatrix<A>> for BaseMatrix<A>
{
    fn add_assign(&mut self, rhs: &BaseMatrix<A>) {
        assert_eq!(self.shape(), rhs.shape());
        self.data += &rhs.data
    }
}
impl<A: BaseType<A> + AddAssign<A> + LinalgScalar + SubAssign<A>> std::ops::Add<BaseMatrix<A>> for BaseMatrix<A>
{
    type Output = BaseMatrix<A>;
    fn add(self, mut rhs: BaseMatrix<A>) -> Self::Output {
        rhs += &self;
        rhs
    }
}

impl<A: BaseType<A> + SubAssign<A> + LinalgScalar + AddAssign<A>> std::ops::SubAssign<BaseMatrix<A>> for BaseMatrix<A>
{
    fn sub_assign(&mut self, rhs: BaseMatrix<A>) {
        assert_eq!(self.shape(), rhs.shape());
        self.data -= &rhs.data
    }
}
impl<A: BaseType<A> + SubAssign<A> + LinalgScalar + AddAssign<A>> std::ops::SubAssign<&BaseMatrix<A>> for BaseMatrix<A>
{
    fn sub_assign(&mut self, rhs: &BaseMatrix<A>) {
        assert_eq!(self.shape(), rhs.shape());
        self.data -= &rhs.data
    }
}
//Subtraction is not implemented yet

impl<A: BaseType<A> + SubAssign<A> + LinalgScalar + AddAssign<A>> std::ops::Index<usize> for BaseMatrix<A>
{
    type Output = [A];
    fn index(&self, index: usize) -> &Self::Output {
        self.get_row(index)
    }
}

//Scalar Multiplication: Undone
impl<A, T> std::ops::MulAssign<T> for BaseMatrix<A>
where A: BaseType<A> + SubAssign<A> + LinalgScalar + AddAssign<A> + MulAssign<A>,
      T: LinalgScalar
{
    fn mul_assign(&mut self, rhs: T) {
        
    }
}
impl<A, T> std::ops::Mul<T> for BaseMatrix<A>
where A: BaseType<A> + SubAssign<A> + LinalgScalar + AddAssign<A> + MulAssign<A>,
      T: LinalgScalar
{
    type Output = BaseMatrix<A>;
    fn mul(self, rhs: T) -> Self::Output {
        self
    }
}

impl<A: BaseType<A> + SubAssign<A> + LinalgScalar + MulAssign<A> + AddAssign<A>> std::ops::MulAssign<BaseMatrix<A>> for BaseMatrix<A>
{
    fn mul_assign(&mut self, rhs: BaseMatrix<A>) {
        self.data *= &rhs.data
    }
}
impl<A: BaseType<A> + SubAssign<A> + LinalgScalar + MulAssign<A> + AddAssign<A>> std::ops::MulAssign<&BaseMatrix<A>> for BaseMatrix<A>
{
    fn mul_assign(&mut self, rhs: &BaseMatrix<A>) {
        self.data *= &rhs.data
    }
}
impl<A: BaseType<A> + SubAssign<A> + LinalgScalar + MulAssign<A> + AddAssign<A>> std::ops::Mul<BaseMatrix<A>> for BaseMatrix<A>
{
    type Output = BaseMatrix<A>;
    fn mul(self, rhs: BaseMatrix<A>) -> Self::Output {
        //Uninmplemented because matrix multiplication is not commutative and i'll have to clone()
        self
    }
}

