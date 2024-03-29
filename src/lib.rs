#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]

pub mod base_array;
pub mod dtype;

//All imports are defined here and made (pub)crate
pub(crate) use core::{num, ops::Range};
pub(crate) use ndarray::{
    iter::LanesIter, Array2, ArrayBase, ArrayView, Axis, Dim, Ix1, Ix2, LinalgScalar, ViewRepr,
};
pub(crate) use std::{
    collections::HashMap,
    fmt::Display,
    fs::OpenOptions,
    io::BufReader,
    iter::{zip, Zip},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    path::PathBuf,
    slice::Iter,
    vec,
};

// fn test_something(){
//     let arr = nalgebra::DMatrix::repeat(100, 100, 4f32);
//     let arr_two = nalgebra::DMatrix::repeat(100, 100, 4f32);
//     let mut alloc: nalgebra::Matrix<f32, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<_, nalgebra::Dyn, nalgebra::Dyn>> = nalgebra::DMatrix::identity(100, 100);
//     alloc.gemm(1f32, &arr, &arr_two, 0f32);
// }

//we will use this to bench the performance of seperate 2d matrices with criterion
//instead of pub(crate)-ing criterion, since we won't use it elsewhere we'll just import it privately
