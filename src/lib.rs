#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]

pub mod base_array;
pub mod dtype;
pub mod linear_models;
pub mod utils;
#[cfg(feature = "gpu_accel")]
pub mod gpu_accel;
//All imports are defined here and made (pub)crate
pub(crate) use core::{
    fmt,
    num::{self, ParseFloatError, ParseIntError},
    ops::Range,
    str::FromStr,
};
pub(crate) use counter::Counter;
pub(crate) use csv::Position;
pub(crate) use float_derive_macros::FloatEq;
pub(crate) use ndarray::{
    iter::{Axes, Indices, LanesIter, LanesMut},
    s, Array2, ArrayBase, ArrayView, ArrayView2, Axis, Dim, IndexLonger, Ix1, Ix2, LinalgScalar,
    ViewRepr,
};
pub(crate) use num_traits::{Float, FromPrimitive, Num, NumCast, ToPrimitive, Zero};
pub(crate) use ordered_float::NotNan;
pub(crate) use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::Display,
    fs::OpenOptions,
    io::{BufReader, Read},
    iter::{zip, Zip},
    mem,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
    path::{Path, PathBuf},
    slice::Iter,
    vec,
};
pub(crate) use thiserror::Error;

// fn test_something(){
//     let arr = nalgebra::DMatrix::repeat(100, 100, 4f32);
//     let arr_two = nalgebra::DMatrix::repeat(100, 100, 4f32);
//     let mut alloc: nalgebra::Matrix<f32, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<_, nalgebra::Dyn, nalgebra::Dyn>> = nalgebra::DMatrix::identity(100, 100);
//     alloc.gemm(1f32, &arr, &arr_two, 0f32);
// }

//we will use this to bench the performance of seperate 2d matrices with criterion
//instead of pub(crate)-ing criterion, since we won't use it elsewhere we'll just import it privately
