use super::*;
use crate::*;

pub const ERR_MSG_INCOMPAT_TYPES: &'static str =
    "Attempt to perform numeric operation on incompatible types!";

#[repr(u8)]
#[derive(Debug, PartialEq, PartialOrd, Clone, FloatEq)]
pub enum DType {
    None,
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Object(Box<String>),
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
                DTypeType::Object => Ok(DType::Object(Box::new(value.to_string()))),
            },
            DType::F64(value) => match rhs {
                DTypeType::None => Ok(DType::None),
                DTypeType::U32 => Ok(DType::U32(*value as u32)),
                DTypeType::U64 => Ok(DType::U64(*value as u64)),
                DTypeType::F32 => Ok(DType::F32(*value as f32)),
                DTypeType::F64 => Ok(DType::F64(*value)),
                DTypeType::Object => Ok(DType::Object(Box::new(value.to_string()))),
            },
            DType::U32(value) => match rhs {
                DTypeType::None => Ok(DType::None),
                DTypeType::U32 => Ok(DType::U32(*value)),
                DTypeType::U64 => Ok(DType::U64(*value as u64)),
                DTypeType::F32 => Ok(DType::F32(*value as f32)),
                DTypeType::F64 => Ok(DType::F64(*value as f64)),
                DTypeType::Object => Ok(DType::Object(Box::new(value.to_string()))),
            },
            DType::U64(value) => match rhs {
                DTypeType::None => Ok(DType::None),
                DTypeType::U32 => Ok(DType::U32(*value as u32)),
                DTypeType::U64 => Ok(DType::U64(*value)),
                DTypeType::F32 => Ok(DType::F32(*value as f32)),
                DTypeType::F64 => Ok(DType::F64(*value as f64)),
                DTypeType::Object => Ok(DType::Object(Box::new(value.to_string()))),
            },
            DType::Object(value) => match rhs {
                DTypeType::None => Ok(DType::None),
                DTypeType::U32 => Ok(DType::U32(str::parse::<u32>(value)?)),
                DTypeType::U64 => Ok(DType::U64(str::parse::<u64>(value)?)),
                DTypeType::F32 => Ok(DType::F32(str::parse::<f32>(value)?)),
                DTypeType::F64 => Ok(DType::F64(str::parse::<f64>(value)?)),
                DTypeType::Object => Ok(DType::Object(Box::new(value.to_string()))),
            },
        }
    }
    pub fn data_type(&self) -> DTypeType {
        DTypeType::from(self)
    }
    #[inline]
    pub fn type_size(&self) -> usize {
        mem::size_of_val(self)
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
            "na" | "NA" | "n/a" | "N/A" | "N/a" | "nan" | "NaN" | "Nan" | "" | "NULL"
        )
    }

    pub fn parse_from_str(input: &str, prefer_precision: bool) -> Self {
        if Self::parses_to_none(input) {
            DType::None
        } else {
            if prefer_precision {
                u64::from_str(input)
                    .map(DType::U64)
                    .or(f64::from_str(input).map(DType::F64))
                    .unwrap_or_else(|_| DType::Object(Box::new(input.to_string())))
            } else {
                u32::from_str(input)
                    .map(DType::U32)
                    .or(f32::from_str(input).map(DType::F32))
                    .or(f64::from_str(input).map(DType::F64))
                    .or(u64::from_str(input).map(DType::U64))
                    .unwrap_or_else(|_| DType::Object(Box::new(input.to_string())))
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
            Object(mut l) => match rhs {
                Object(r) => {
                    l.push_str(&r);
                    Object(l)
                }
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
            Object(mut l) => match rhs {
                Object(r) => {
                    l.push_str(r);
                    Object(l)
                }
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
                Object(r) => Object(Box::new(l.as_ref().clone() + r)),
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
            Object(mut l) => match rhs {
                Object(r) => {
                    l.push_str(r);
                    Object(l)
                }
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
                Object(r) => Object(Box::new(l.as_ref().clone() + r)),
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
            Object(_) => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
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
            *self = unsafe { (self as *const DType).as_ref().unwrap().clone() * &rhs }
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
            DType::None => write!(f, "NAN"),
            DType::F32(val_) => write!(f, "{val_}"),
            DType::F64(val_) => write!(f, "{val_}"),
            DType::U32(val_) => write!(f, "{val_}"),
            DType::U64(val_) => write!(f, "{val_}"),
            DType::Object(val) => write!(f, "{val}"),
        }
    }
}

impl From<u32> for DType {
    fn from(value: u32) -> Self {
        DType::U32(value)
    }
}

impl From<u64> for DType {
    fn from(value: u64) -> Self {
        DType::U64(value)
    }
}

impl From<f32> for DType {
    fn from(value: f32) -> Self {
        if value.is_nan() | value.is_infinite() | value.is_subnormal() {
            return DType::None;
        }
        DType::F32(value)
    }
}

impl From<f64> for DType {
    fn from(value: f64) -> Self {
        if value.is_nan() | value.is_infinite() | value.is_subnormal() {
            return DType::None;
        }
        DType::F64(value)
    }
}

impl From<usize> for DType {
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

impl From<Box<String>> for DType {
    fn from(value: Box<String>) -> Self {
        DType::Object(value)
    }
}

impl From<&str> for DType {
    fn from(value: &str) -> Self {
        DType::Object(Box::new(value.to_string()))
    }
}

impl Zero for DType {
    fn is_zero(&self) -> bool {
        match self {
            DType::None => false,
            DType::F32(val) => *val == 0f32,
            DType::F64(val) => *val == 0f64,
            DType::U32(val) => *val == 0,
            DType::U64(val) => *val == 0,
            DType::Object(_) => false,
        }
    }

    fn zero() -> Self {
        0f64.into()
    }
}

impl Neg for DType {
    type Output = DType;
    fn neg(self) -> Self::Output {
        match self {
            DType::None => DType::None,
            DType::F32(var) => DType::F32(-var),
            DType::F64(var) => DType::F64(-var),
            DType::U32(var) => DType::F32(-(var as f32)),
            DType::U64(var) => DType::F64(-(var as f64)),
            _ => panic!("{}", ERR_MSG_INCOMPAT_TYPES),
        }
    }
}

impl Ord for DType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let this_as_float = self.to_f64();
        let other_as_float = other.to_f64();
        if this_as_float.is_some() && other_as_float.is_some() {
            return this_as_float.unwrap().total_cmp(&other_as_float.unwrap());
        }
        if (*self) == DType::None {
            if (*other) == DType::None {
                return Ordering::Equal;
            } else {
                return Ordering::Greater;
            }
        }
        if (*other) == DType::None {
            return Ordering::Less;
        }
        //both dtypes are now surely Object
        if let DType::Object(this_as_str) = self {
            if let DType::Object(other_as_str) = other {
                this_as_str.cmp(&other_as_str)
            } else {
                Ordering::Equal
            } //this should never be hit
        } else {
            Ordering::Equal
        }
    }
}

impl ToPrimitive for DType {
    fn to_u64(&self) -> Option<u64> {
        match self {
            DType::F32(var) => {
                if *var < 0.0f32 {
                    Option::None
                } else {
                    Option::Some(*var as u64)
                }
            }
            DType::F64(var) => {
                if *var < 0.0f64 {
                    Option::None
                } else {
                    Option::Some(*var as u64)
                }
            }
            DType::U32(var) => Option::Some(*var as u64),
            DType::U64(var) => Option::Some(*var as u64),
            _ => Option::None,
        }
    }
    fn to_i64(&self) -> Option<i64> {
        match self {
            DType::F32(var) => Option::Some(*var as i64),
            DType::F64(var) => Option::Some(*var as i64),
            DType::U32(var) => Option::Some(*var as i64),
            DType::U64(var) => Option::Some(*var as i64),
            _ => Option::None,
        }
    }
    fn to_isize(&self) -> Option<isize> {
        match self {
            DType::F32(var) => isize::try_from((*var) as i64).ok(),
            DType::F64(var) => isize::try_from((*var) as i64).ok(),
            DType::U32(var) => isize::try_from(*var).ok(),
            DType::U64(var) => isize::try_from(*var).ok(),
            _ => Option::None,
        }
    }
    fn to_i8(&self) -> Option<i8> {
        match self {
            DType::F32(var) => i8::try_from((*var) as i64).ok(),
            DType::F64(var) => i8::try_from((*var) as i64).ok(),
            DType::U32(var) => i8::try_from(*var).ok(),
            DType::U64(var) => i8::try_from(*var).ok(),
            _ => Option::None,
        }
    }
    fn to_i16(&self) -> Option<i16> {
        match self {
            DType::F32(var) => i16::try_from((*var) as i64).ok(),
            DType::F64(var) => i16::try_from((*var) as i64).ok(),
            DType::U32(var) => i16::try_from(*var).ok(),
            DType::U64(var) => i16::try_from(*var).ok(),
            _ => Option::None,
        }
    }
    fn to_i32(&self) -> Option<i32> {
        match self {
            DType::F32(var) => i32::try_from((*var) as i64).ok(),
            DType::F64(var) => i32::try_from((*var) as i64).ok(),
            DType::U32(var) => i32::try_from(*var).ok(),
            DType::U64(var) => i32::try_from(*var).ok(),
            _ => Option::None,
        }
    }
    fn to_i128(&self) -> Option<i128> {
        match self {
            DType::F32(var) => Option::Some(*var as i128),
            DType::F64(var) => Option::Some(*var as i128),
            DType::U32(var) => Option::Some(*var as i128),
            DType::U64(var) => Option::Some(*var as i128),
            _ => Option::None,
        }
    }
    fn to_usize(&self) -> Option<usize> {
        match self {
            DType::F32(var) => {
                if *var < 0.0f32 {
                    Option::None
                } else {
                    usize::try_from((*var) as u64).ok()
                }
            }
            DType::F64(var) => {
                if *var < 0.0f64 {
                    Option::None
                } else {
                    usize::try_from((*var) as u64).ok()
                }
            }
            DType::U32(var) => usize::try_from(*var).ok(),
            DType::U64(var) => usize::try_from(*var).ok(),
            _ => Option::None,
        }
    }
    fn to_u8(&self) -> Option<u8> {
        match self {
            DType::F32(var) => {
                if *var < 0.0f32 {
                    Option::None
                } else {
                    u8::try_from((*var) as u64).ok()
                }
            }
            DType::F64(var) => {
                if *var < 0.0f64 {
                    Option::None
                } else {
                    u8::try_from((*var) as u64).ok()
                }
            }
            DType::U32(var) => u8::try_from(*var).ok(),
            DType::U64(var) => u8::try_from(*var).ok(),
            _ => Option::None,
        }
    }
    fn to_u16(&self) -> Option<u16> {
        match self {
            DType::F32(var) => {
                if *var < 0.0f32 {
                    Option::None
                } else {
                    u16::try_from((*var) as u64).ok()
                }
            }
            DType::F64(var) => {
                if *var < 0.0f64 {
                    Option::None
                } else {
                    u16::try_from((*var) as u64).ok()
                }
            }
            DType::U32(var) => u16::try_from(*var).ok(),
            DType::U64(var) => u16::try_from(*var).ok(),
            _ => Option::None,
        }
    }
    fn to_u32(&self) -> Option<u32> {
        match self {
            DType::F32(var) => {
                if *var < 0.0f32 {
                    Option::None
                } else {
                    u32::try_from((*var) as u64).ok()
                }
            }
            DType::F64(var) => {
                if *var < 0.0f64 {
                    Option::None
                } else {
                    u32::try_from((*var) as u64).ok()
                }
            }
            DType::U32(var) => u32::try_from(*var).ok(),
            DType::U64(var) => u32::try_from(*var).ok(),
            _ => Option::None,
        }
    }
    fn to_u128(&self) -> Option<u128> {
        match self {
            DType::F32(var) => {
                if *var < 0.0f32 {
                    Option::None
                } else {
                    Option::Some(*var as u128)
                }
            }
            DType::F64(var) => {
                if *var < 0.0f64 {
                    Option::None
                } else {
                    Option::Some(*var as u128)
                }
            }
            DType::U32(var) => Option::Some(*var as u128),
            DType::U64(var) => Option::Some(*var as u128),
            _ => Option::None,
        }
    }
    fn to_f32(&self) -> Option<f32> {
        match self {
            DType::F32(var) => Option::Some(*var as f32),
            DType::F64(var) => Option::Some(*var as f32),
            DType::U32(var) => Option::Some(*var as f32),
            DType::U64(var) => Option::Some(*var as f32),
            _ => Option::None,
        }
    }
    fn to_f64(&self) -> Option<f64> {
        match self {
            DType::F32(var) => Option::Some(*var as f64),
            DType::F64(var) => Option::Some(*var as f64),
            DType::U32(var) => Option::Some(*var as f64),
            DType::U64(var) => Option::Some(*var as f64),
            _ => Option::None,
        }
    }
}

impl NumCast for DType {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        Some(DType::F64(n.to_f64().unwrap()))
    }
}

impl Default for DType {
    fn default() -> Self {
        DType::F64(0f64)
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
