use micromath::F32;

use crate::*;
use core::{
    fmt,
    num::{ParseFloatError, ParseIntError},
    str::FromStr,
};
use std::mem;
use thiserror::Error;

const ERR_MSG_INCOMPAT_TYPES: &'static str = "Attempt to perform numeric operation on imcompatible types!";

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub enum DType {
    None,
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Object(String),
}

impl DType {
    // Prime candidate for marcroization 👇
    pub fn cast(&self, rhs: DTypeType) -> Result<Self, Error> {
        match self {
            //None type is not convertible. This will return none by default.
            DType::None => Ok(DType::None),
            DType::F32(value) => match rhs {
                DTypeType::None => Ok(DType::None),
                DTypeType::U32 => Ok(DType::U32(*value as u32)),
                DTypeType::U64 => Ok(DType::U64(*value as u64)),
                DTypeType::F32 => Ok(DType::F32(*value)),
                DTypeType::F64 => Ok(DType::F64(*value as f64)),
                DTypeType::Object => Ok(DType::Object(value.to_string())),
            },
            DType::F64(value) => match rhs {
                DTypeType::None => Ok(DType::None),
                DTypeType::U32 => Ok(DType::U32(*value as u32)),
                DTypeType::U64 => Ok(DType::U64(*value as u64)),
                DTypeType::F32 => Ok(DType::F32(*value as f32)),
                DTypeType::F64 => Ok(DType::F64(*value)),
                DTypeType::Object => Ok(DType::Object(value.to_string())),
            },
            DType::U32(value) => match rhs {
                DTypeType::None => Ok(DType::None),
                DTypeType::U32 => Ok(DType::U32(*value)),
                DTypeType::U64 => Ok(DType::U64(*value as u64)),
                DTypeType::F32 => Ok(DType::F32(*value as f32)),
                DTypeType::F64 => Ok(DType::F64(*value as f64)),
                DTypeType::Object => Ok(DType::Object(value.to_string())),
            },
            DType::U64(value) => match rhs {
                DTypeType::None => Ok(DType::None),
                DTypeType::U32 => Ok(DType::U32(*value as u32)),
                DTypeType::U64 => Ok(DType::U64(*value)),
                DTypeType::F32 => Ok(DType::F32(*value as f32)),
                DTypeType::F64 => Ok(DType::F64(*value as f64)),
                DTypeType::Object => Ok(DType::Object(value.to_string())),
            },
            DType::Object(value) => match rhs {
                DTypeType::None => Ok(DType::None),
                DTypeType::U32 => Ok(DType::U32(str::parse::<u32>(value)?)),
                DTypeType::U64 => Ok(DType::U64(str::parse::<u64>(value)?)),
                DTypeType::F32 => Ok(DType::F32(str::parse::<f32>(value)?)),
                DTypeType::F64 => Ok(DType::F64(str::parse::<f64>(value)?)),
                DTypeType::Object => Ok(DType::Object(value.to_string())),
            },
        }
    }

    pub fn data_type(&self) -> DTypeType {
        DTypeType::from(self)
    }

    #[inline]
    pub fn type_size(&self) -> usize {
        match self {
            DType::None => mem::size_of::<DType>(),
            DType::F32(_) => mem::size_of::<DType>() + 4,
            DType::F64(_) => mem::size_of::<DType>() + 8,
            DType::U32(_) => mem::size_of::<DType>() + 4,
            DType::U64(_) => mem::size_of::<DType>() + 8,
            DType::Object(data) => mem::size_of::<DType>() + data.len(),
        }
    }

    pub fn abs(&mut self) {
        match self {
            DType::None => {}
            DType::F32(val) => *val = val.abs(),
            DType::F64(val) => *val = val.abs(),
            DType::U32(_) => {}
            DType::U64(_) => {}
            DType::Object(_) => {}
        }
    }

    pub fn parses_to_none(input: &str) -> bool {
        matches!(
            input,
            "na" | "NA" | "n/a" | "N/A" | "N/a" | "nan" | "NaN" | "Nan"
        )
    }

    pub fn parse_from_str(input: &str, prefer_precision: bool) -> Self {
        if Self::parses_to_none(input) {
            DType::None
        } else {
            if prefer_precision {
                f64::from_str(input)
                    .map(DType::F64)
                    .or(f32::from_str(input).map(DType::F32))
                    .or(u64::from_str(input).map(DType::U64))
                    .or(u32::from_str(input).map(DType::U32))
                    .unwrap_or_else(|_| DType::Object(input.to_string()))
            } else {
                f32::from_str(input)
                    .map(DType::F32)
                    .or(f64::from_str(input).map(DType::F64))
                    .or(u32::from_str(input).map(DType::U32))
                    .or(u64::from_str(input).map(DType::U64))
                    .unwrap_or_else(|_| DType::Object(input.to_string()))
            }
        }
    }
}

impl Add<DType> for DType {
    type Output = DType;

    fn add(self, rhs: DType) -> Self::Output {
        use DType::*;

        match self {
            // There has got to be a better way.
            None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            U32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U32(l + r),
                U64(r) => U64(l as u64 + r),
                F32(r) => F32(l as f32 + r),
                F64(r) => F64(l as f64 + r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            U64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U64(l + r as u64),
                U64(r) => U64(l + r),
                F32(r) => F32(l as f32 + r),
                F64(r) => F64(l as f64 + r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F32(l + r as f32),
                U64(r) => F32(l + r as f32),
                F32(r) => F32(l + r),
                F64(r) => F64(l as f64 + r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F64(l + r as f64),
                U64(r) => F64(l + r as f64),
                F32(r) => F64(l + r as f64),
                F64(r) => F64(l + r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            Object(l) => match rhs {
                Object(r) => Object(l + &r),
                _ => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
        }
    }
}

impl Add<&DType> for DType {
    type Output = DType;

    fn add(self, rhs: &DType) -> Self::Output {
        use DType::*;

        match self {
            // There has got to be a better way.
            None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            U32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U32(l + r),
                U64(r) => U64(l as u64 + r),
                F32(r) => F32(l as f32 + r),
                F64(r) => F64(l as f64 + r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            U64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U64(l + *r as u64),
                U64(r) => U64(l + r),
                F32(r) => F32(l as f32 + r),
                F64(r) => F64(l as f64 + r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F32(l + *r as f32),
                U64(r) => F32(l + *r as f32),
                F32(r) => F32(l + r),
                F64(r) => F64(l as f64 + r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F64(l + *r as f64),
                U64(r) => F64(l + *r as f64),
                F32(r) => F64(l + *r as f64),
                F64(r) => F64(l + r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            Object(l) => match rhs {
                Object(r) => Object(l + r),
                _ => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
        }
    }
}

impl Add<&DType> for &DType {
    type Output = DType;

    fn add(self, rhs: &DType) -> Self::Output {
        use DType::*;

        match self {
            None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            U32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U32(l + r),
                U64(r) => U64(*l as u64 + r),
                F32(r) => F32(*l as f32 + r),
                F64(r) => F64(*l as f64 + r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            U64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U64(l + *r as u64),
                U64(r) => U64(l + r),
                F32(r) => F32(*l as f32 + r),
                F64(r) => F64(*l as f64 + r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F32(l + *r as f32),
                U64(r) => F32(l + *r as f32),
                F32(r) => F32(l + r),
                F64(r) => F64(*l as f64 + r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F64(l + *r as f64),
                U64(r) => F64(l + *r as f64),
                F32(r) => F64(l + *r as f64),
                F64(r) => F64(l + r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            Object(l) => match rhs {
                Object(r) => Object(l.clone() + r),
                _ => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
        }
    }
}

impl Sub<&DType> for DType {
    type Output = DType;

    fn sub(self, rhs: &DType) -> Self::Output {
        use DType::*;

        match self {
            None => self,
            U32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U32(l - r),
                U64(r) => U64(l as u64 - r),
                F32(r) => F32(l as f32 - r),
                F64(r) => F64(l as f64 - r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            U64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U64(l - *r as u64),
                U64(r) => U64(l - r),
                F32(r) => F32(l as f32 - r),
                F64(r) => F64(l as f64 - r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F32(l - *r as f32),
                U64(r) => F32(l - *r as f32),
                F32(r) => F32(l - r),
                F64(r) => F64(l as f64 - r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F64(l - *r as f64),
                U64(r) => F64(l - *r as f64),
                F32(r) => F64(l - *r as f64),
                F64(r) => F64(l - r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            Object(l) => Object(l),
        }
    }
}

impl Sub<&DType> for &DType {
    type Output = DType;

    fn sub(self, rhs: &DType) -> Self::Output {
        use DType::*;

        match self {
            None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            U32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U32(l - r),
                U64(r) => U64(*l as u64 - r),
                F32(r) => F32(*l as f32 - r),
                F64(r) => F64(*l as f64 - r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            U64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U64(l - *r as u64),
                U64(r) => U64(l - r),
                F32(r) => F32(*l as f32 - r),
                F64(r) => F64(*l as f64 - r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F32(l - *r as f32),
                U64(r) => F32(l - *r as f32),
                F32(r) => F32(l - r),
                F64(r) => F64(*l as f64 - r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F64(l - *r as f64),
                U64(r) => F64(l - *r as f64),
                F32(r) => F64(l - *r as f64),
                F64(r) => F64(l - r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            Object(l) => Object(l.clone()),
        }
    }
}

impl Mul<&DType> for DType {
    type Output = DType;

    fn mul(self, rhs: &DType) -> Self::Output {
        use DType::*;

        match self {
            None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            U32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U32(l * r),
                U64(r) => U64(l as u64 * r),
                F32(r) => F32(l as f32 * r),
                F64(r) => F64(l as f64 * r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            U64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U64(l * *r as u64),
                U64(r) => U64(l * r),
                F32(r) => F32(l as f32 * r),
                F64(r) => F64(l as f64 * r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F32(l * *r as f32),
                U64(r) => F32(l * *r as f32),
                F32(r) => F32(l * r),
                F64(r) => F64(l as f64 * r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F64(l * *r as f64),
                U64(r) => F64(l * *r as f64),
                F32(r) => F64(l * *r as f64),
                F64(r) => F64(l * r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            Object(l) => match rhs {
                Object(r) => Object(l + r),
                _ => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
        }
    }
}

impl Mul<&DType> for &DType {
    type Output = DType;

    fn mul(self, rhs: &DType) -> Self::Output {
        use DType::*;

        match self {
            None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            U32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U32(l * r),
                U64(r) => U64(*l as u64 * r),
                F32(r) => F32(*l as f32 * r),
                F64(r) => F64(*l as f64 * r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            U64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U64(l * *r as u64),
                U64(r) => U64(l * r),
                F32(r) => F32(*l as f32 * r),
                F64(r) => F64(*l as f64 * r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F32(l * *r as f32),
                U64(r) => F32(l * *r as f32),
                F32(r) => F32(l * r),
                F64(r) => F64(*l as f64 * r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F64(l * *r as f64),
                U64(r) => F64(l * *r as f64),
                F32(r) => F64(l * *r as f64),
                F64(r) => F64(l * r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            Object(l) => match rhs {
                Object(r) => Object(l.clone() + r),
                _ => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
        }
    }
}

impl Div<&DType> for DType {
    type Output = DType;

    fn div(self, rhs: &DType) -> Self::Output {
        use DType::*;

        match self {
            None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            U32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U32(l / r),
                U64(r) => U64(l as u64 / r),
                F32(r) => F32(l as f32 / r),
                F64(r) => F64(l as f64 / r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            U64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U64(l / *r as u64),
                U64(r) => U64(l / r),
                F32(r) => F32(l as f32 / r),
                F64(r) => F64(l as f64 / r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F32(l / *r as f32),
                U64(r) => F32(l / *r as f32),
                F32(r) => F32(l / r),
                F64(r) => F64(l as f64 / r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F64(l / *r as f64),
                U64(r) => F64(l / *r as f64),
                F32(r) => F64(l / *r as f64),
                F64(r) => F64(l / r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            Object(l) => Object(l),
        }
    }
}

impl Div<DType> for DType {
    type Output = DType;

    fn div(self, rhs: DType) -> Self::Output {
        use DType::*;

        match self {
            None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            U32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U32(l / r),
                U64(r) => U64(l as u64 / r),
                F32(r) => F32(l as f32 / r),
                F64(r) => F64(l as f64 / r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            U64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U64(l / r as u64),
                U64(r) => U64(l / r),
                F32(r) => F32(l as f32 / r),
                F64(r) => F64(l as f64 / r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F32(l / r as f32),
                U64(r) => F32(l / r as f32),
                F32(r) => F32(l / r),
                F64(r) => F64(l as f64 / r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F64(l / r as f64),
                U64(r) => F64(l / r as f64),
                F32(r) => F64(l / r as f64),
                F64(r) => F64(l / r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            Object(l) => Object(l),
        }
    }
}



impl Div<&DType> for &DType {
    type Output = DType;

    fn div(self, rhs: &DType) -> Self::Output {
        use DType::*;

        match self {
            None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            U32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U32(l / r),
                U64(r) => U64(*l as u64 / r),
                F32(r) => F32(*l as f32 / r),
                F64(r) => F64(*l as f64 / r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            U64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => U64(l / *r as u64),
                U64(r) => U64(l / r),
                F32(r) => F32(*l as f32 / r),
                F64(r) => F64(*l as f64 / r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F32(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F32(l / *r as f32),
                U64(r) => F32(l / *r as f32),
                F32(r) => F32(l / r),
                F64(r) => F64(*l as f64 / r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            F64(l) => match rhs {
                None => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
                U32(r) => F64(l / *r as f64),
                U64(r) => F64(l / *r as f64),
                F32(r) => F64(l / *r as f64),
                F64(r) => F64(l / r),
                Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
            },
            Object(l) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
        }
    }
}

impl AddAssign<DType> for DType {
    fn add_assign(&mut self, rhs: DType) {
        use DType::*;

        if let Object(l) = self {
            if let Object(r) = &rhs {
                l.push_str(r);
            }
            // else, no change
        } else {
            *self = &*self + &rhs;
        }
    }
}

impl AddAssign<&DType> for DType {
    fn add_assign(&mut self, rhs: &DType) {
        use DType::*;

        if let Object(l) = self {
            if let Object(r) = rhs {
                l.push_str(r);
            }
            // else, no change
        } else {
            *self = &*self + rhs;
        }
    }
}

impl SubAssign<DType> for DType {
    fn sub_assign(&mut self, rhs: DType) {
        use DType::*;

        match self {
            Object(_) => { /* no change */ }
            _ => *self = &*self - &rhs,
        }
    }
}

impl SubAssign<&DType> for DType {
    fn sub_assign(&mut self, rhs: &DType) {
        use DType::*;

        match self {
            Object(_) => { /* no change */ }
            _ => *self = &*self - rhs,
        }
    }
}

impl MulAssign<DType> for DType {
    fn mul_assign(&mut self, rhs: DType) {
        use DType::*;

        if let Object(l) = self {
            if let Object(r) = &rhs {
                l.push_str(r);
            }
            // else, no change
        } else {
            *self = &*self * &rhs;
        }
    }
}

impl MulAssign<&DType> for DType {
    fn mul_assign(&mut self, rhs: &DType) {
        use DType::*;

        if let Object(l) = self {
            if let Object(r) = rhs {
                l.push_str(r);
            }
            // else, no change
        } else {
            *self = &*self * rhs;
        }
    }
}

impl DivAssign<DType> for DType {
    fn div_assign(&mut self, rhs: DType) {
        use DType::*;

        match self {
            Object(_) => { /* no change */ }
            _ => *self = &*self / &rhs,
        }
    }
}

impl DivAssign<&DType> for DType {
    fn div_assign(&mut self, rhs: &DType) {
        use DType::*;

        match self {
            Object(_) => { /* no change */ }
            _ => *self = &*self / rhs,
        }
    }
}

impl Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            //maybe i was a little too hasty with this...
            //ugh i'll have to manually do this there, then
            DType::None => write!(f, "NAN"),
            DType::F32(val_) => write!(f, "{val_}"),
            DType::F64(val_) => write!(f, "{val_}"),
            DType::U32(val_) => write!(f, "{val_}"),
            DType::U64(val_) => write!(f, "{val_}"),
            DType::Object(val) => write!(f, "{val}"),
        }
    }
}

impl From<u32> for DType{
    fn from(value: u32) -> Self {
        DType::U32(value)
    }
}

impl From<u64> for DType{
    fn from(value: u64) -> Self {
        DType::U64(value)
    }
}

impl From<f32> for DType{
    fn from(value: f32) -> Self {
        if value.is_nan() | value.is_infinite() | value.is_subnormal(){
            return DType::None
       }
       DType::F32(value)
    }
}

impl From<f64> for DType{
    fn from(value: f64) -> Self {
       if value.is_nan() | value.is_infinite() | value.is_subnormal(){
            return DType::None
       }
       DType::F64(value)
    }
}

impl From<usize> for DType{
    #[cfg(target_pointer_width = "16")]
    fn from(value: usize) -> Self {
        DType::U32(value as u32)
    }

    #[cfg(target_pointer_width = "32")]
    fn from(value: usize) -> Self {
        DType::U32(value)
    }

    #[cfg(target_pointer_width = "64")]
    fn from(value: usize) -> Self {
        DType::U64(value as u64) //should fit perfectly
    }
}

impl From<String> for DType{
    fn from(value: String) -> Self {
        DType::Object(value)
    }
}

impl From<&str> for DType{
    fn from(value: &str) -> Self {
        DType::Object(value.to_string())
    }
}

impl Zero for DType{
    fn is_zero(&self) -> bool {
        match self{
            DType::None => false,
            DType::F32(val) => *val == 0f32,
            DType::F64(val) => *val == 0f64,
            DType::U32(val) => *val == 0,
            DType::U64(val) => *val == 0,
            DType::Object(_) => false
        }
    }

    fn zero() -> Self {
        0f64.into()
    }
}



#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    ParseInt(#[from] core::num::ParseIntError),
    #[error(transparent)]
    ParseFloat(#[from] core::num::ParseFloatError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DTypeType {
    None,
    U32,
    U64,
    F32,
    F64,
    Object,
}

impl DTypeType {
    pub fn display_str(&self) -> &'static str {
        match self {
            DTypeType::None => "None",
            DTypeType::U32 => "u32",
            DTypeType::U64 => "u64",
            DTypeType::F32 => "f32",
            DTypeType::F64 => "f64",
            DTypeType::Object => "object",
        }
    }
}

impl Display for DTypeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.display_str())?;

        Ok(())
    }
}

impl From<&DType> for DTypeType {
    fn from(value: &DType) -> Self {
        match value {
            DType::None => DTypeType::None,
            DType::U32(_) => DTypeType::U32,
            DType::U64(_) => DTypeType::U64,
            DType::F32(_) => DTypeType::F32,
            DType::F64(_) => DTypeType::F64,
            DType::Object(_) => DTypeType::Object,
        }
    }
}
