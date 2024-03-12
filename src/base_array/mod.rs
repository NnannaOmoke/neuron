
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

#[derive(Debug, Copy, PartialEq, PartialOrd, Hash, Clone)]
pub enum DType<'a >{
    Nothing,
    U32, 
    U64,
    F32, 
    F64, 
    Object(&'a str),//lmao
}

pub(crate) trait BaseType<T>
where T: Sized + Copy {}

#[repr(C)]
pub(crate) struct BaseMatrix<A: BaseType<A> + Copy>{
    data: Array2<A>
}

impl<A: BaseType<A> + Copy> BaseMatrix<A>{
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


#[repr(C)]
pub struct Dataset<'a, X: BaseType<X> + Copy, Y: BaseType<Y> + Copy>{
    x_data: BaseMatrix<X>,
    y_data: BaseMatrix<Y>,
    fieldnames: Option<HashMap<String, int>>,
    mean: Option<&'a [double]>,
    std: Option<&'a [double]>,
    max: Option<&'a [X]>,
    min: Option<&'a [X]>,
}

impl <'a, X: BaseType<X> + Copy, Y: BaseType<Y> + Copy>  Dataset<'a, X, Y>{
    pub fn from_base_matrix(x: BaseMatrix<X>, y: BaseMatrix<Y>, cached: bool) -> Self{
        if cached{
            //compute all the relevant stats for all the relevant cols(features), on creation
            //if the type associated with a col is a String/object, ignore it
            //which will be complete when the iterators for BaseMatrix is comlete 
            todo!()
        }
        todo!()
    }
    pub fn from_csv(fname: PathBuf, sep: &str, ) -> Self{
        todo!()
    }

}