#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]
//All imports are defined here and made (pub)crate
pub(crate) use std::{
    fs::OpenOptions, 
    path::PathBuf,
    io::BufReader,
    collections::HashMap,
};
pub(crate) use ndarray::{Array2, Ix2};
use nalgebra;

// fn test_something(){
//     let arr = nalgebra::DMatrix::repeat(100, 100, 4f32);
//     let arr_two = nalgebra::DMatrix::repeat(100, 100, 4f32);
//     let mut alloc: nalgebra::Matrix<f32, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<_, nalgebra::Dyn, nalgebra::Dyn>> = nalgebra::DMatrix::identity(100, 100);
//     alloc.gemm(1f32, &arr, &arr_two, 0f32);
// }


//we will use this to bench the performance of seperate 2d matrices with criterion
//instead of pub(crate)-ing criterion, since we won't use it elsewhere we'll just import it privately
pub mod dtype;
mod base_array;



