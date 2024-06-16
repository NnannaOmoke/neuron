#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]
#![allow(suspicious_double_ref_op)]

pub mod base_array;
pub mod dtype;
#[cfg(feature = "gpu_accel")]
pub mod gpu_accel;
pub mod linear_models;
pub mod utils;

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

fn test_something() {}
