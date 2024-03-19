use crate::*;
use core::fmt;
use thiserror::Error;

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

    pub fn type_size(&self) -> usize {
        match self {
            DType::None => 0,
            DType::F32(_) => 4,
            DType::F64(_) => 8,
            DType::U32(_) => 4,
            DType::U64(_) => 8,
            DType::Object(data) => data.len(),
        }
    }
}

impl Add<&DType> for DType {
    type Output = DType;

    fn add(self, rhs: &DType) -> Self::Output {
        use DType::*;

        match self {
            // There has got to be a better way.
            None => rhs.clone(),
            U32(l) => match rhs {
                None => U32(l),
                U32(r) => U32(l + r),
                U64(r) => U64(l as u64 + r),
                F32(r) => F32(l as f32 + r),
                F64(r) => F64(l as f64 + r),
                Object(_) => U32(l),
            },
            U64(l) => match rhs {
                None => U64(l),
                U32(r) => U64(l + *r as u64),
                U64(r) => U64(l + r),
                F32(r) => F32(l as f32 + r),
                F64(r) => F64(l as f64 + r),
                Object(_) => U64(l),
            },
            F32(l) => match rhs {
                None => F32(l),
                U32(r) => F32(l + *r as f32),
                U64(r) => F32(l + *r as f32),
                F32(r) => F32(l + r),
                F64(r) => F64(l as f64 + r),
                Object(_) => F32(l),
            },
            F64(l) => match rhs {
                None => F64(l),
                U32(r) => F64(l + *r as f64),
                U64(r) => F64(l + *r as f64),
                F32(r) => F64(l + *r as f64),
                F64(r) => F64(l + r),
                Object(_) => F64(l),
            },
            Object(l) => match rhs {
                Object(r) => Object(l + r),
                _ => Object(l),
            },
        }
    }
}

impl Add<&DType> for &DType {
    type Output = DType;

    fn add(self, rhs: &DType) -> Self::Output {
        use DType::*;

        match self {
            None => rhs.clone(),
            U32(l) => match rhs {
                None => U32(*l),
                U32(r) => U32(l + r),
                U64(r) => U64(*l as u64 + r),
                F32(r) => F32(*l as f32 + r),
                F64(r) => F64(*l as f64 + r),
                Object(_) => U32(*l),
            },
            U64(l) => match rhs {
                None => U64(*l),
                U32(r) => U64(l + *r as u64),
                U64(r) => U64(l + r),
                F32(r) => F32(*l as f32 + r),
                F64(r) => F64(*l as f64 + r),
                Object(_) => U64(*l),
            },
            F32(l) => match rhs {
                None => F32(*l),
                U32(r) => F32(l + *r as f32),
                U64(r) => F32(l + *r as f32),
                F32(r) => F32(l + r),
                F64(r) => F64(*l as f64 + r),
                Object(_) => F32(*l),
            },
            F64(l) => match rhs {
                None => F64(*l),
                U32(r) => F64(l + *r as f64),
                U64(r) => F64(l + *r as f64),
                F32(r) => F64(l + *r as f64),
                F64(r) => F64(l + r),
                Object(_) => F64(*l),
            },
            Object(l) => match rhs {
                Object(r) => Object(l.clone() + r),
                _ => Object(l.clone()),
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
                None => U32(l),
                U32(r) => U32(l - r),
                U64(r) => U64(l as u64 - r),
                F32(r) => F32(l as f32 - r),
                F64(r) => F64(l as f64 - r),
                Object(_) => U32(l),
            },
            U64(l) => match rhs {
                None => U64(l),
                U32(r) => U64(l - *r as u64),
                U64(r) => U64(l - r),
                F32(r) => F32(l as f32 - r),
                F64(r) => F64(l as f64 - r),
                Object(_) => U64(l),
            },
            F32(l) => match rhs {
                None => F32(l),
                U32(r) => F32(l - *r as f32),
                U64(r) => F32(l - *r as f32),
                F32(r) => F32(l - r),
                F64(r) => F64(l as f64 - r),
                Object(_) => F32(l),
            },
            F64(l) => match rhs {
                None => F64(l),
                U32(r) => F64(l - *r as f64),
                U64(r) => F64(l - *r as f64),
                F32(r) => F64(l - *r as f64),
                F64(r) => F64(l - r),
                Object(_) => F64(l),
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
            None => None,
            U32(l) => match rhs {
                None => U32(*l),
                U32(r) => U32(l - r),
                U64(r) => U64(*l as u64 - r),
                F32(r) => F32(*l as f32 - r),
                F64(r) => F64(*l as f64 - r),
                Object(_) => U32(*l),
            },
            U64(l) => match rhs {
                None => U64(*l),
                U32(r) => U64(l - *r as u64),
                U64(r) => U64(l - r),
                F32(r) => F32(*l as f32 - r),
                F64(r) => F64(*l as f64 - r),
                Object(_) => U64(*l),
            },
            F32(l) => match rhs {
                None => F32(*l),
                U32(r) => F32(l - *r as f32),
                U64(r) => F32(l - *r as f32),
                F32(r) => F32(l - r),
                F64(r) => F64(*l as f64 - r),
                Object(_) => F32(*l),
            },
            F64(l) => match rhs {
                None => F64(*l),
                U32(r) => F64(l - *r as f64),
                U64(r) => F64(l - *r as f64),
                F32(r) => F64(l - *r as f64),
                F64(r) => F64(l - r),
                Object(_) => F64(*l),
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
            None => rhs.clone(),
            U32(l) => match rhs {
                None => U32(l),
                U32(r) => U32(l * r),
                U64(r) => U64(l as u64 * r),
                F32(r) => F32(l as f32 * r),
                F64(r) => F64(l as f64 * r),
                Object(_) => U32(l),
            },
            U64(l) => match rhs {
                None => U64(l),
                U32(r) => U64(l * *r as u64),
                U64(r) => U64(l * r),
                F32(r) => F32(l as f32 * r),
                F64(r) => F64(l as f64 * r),
                Object(_) => U64(l),
            },
            F32(l) => match rhs {
                None => F32(l),
                U32(r) => F32(l * *r as f32),
                U64(r) => F32(l * *r as f32),
                F32(r) => F32(l * r),
                F64(r) => F64(l as f64 * r),
                Object(_) => F32(l),
            },
            F64(l) => match rhs {
                None => F64(l),
                U32(r) => F64(l * *r as f64),
                U64(r) => F64(l * *r as f64),
                F32(r) => F64(l * *r as f64),
                F64(r) => F64(l * r),
                Object(_) => F64(l),
            },
            Object(l) => match rhs {
                Object(r) => Object(l + r),
                _ => Object(l),
            },
        }
    }
}

impl Mul<&DType> for &DType {
    type Output = DType;

    fn mul(self, rhs: &DType) -> Self::Output {
        use DType::*;

        match self {
            None => rhs.clone(),
            U32(l) => match rhs {
                None => U32(*l),
                U32(r) => U32(l * r),
                U64(r) => U64(*l as u64 * r),
                F32(r) => F32(*l as f32 * r),
                F64(r) => F64(*l as f64 * r),
                Object(_) => U32(*l),
            },
            U64(l) => match rhs {
                None => U64(*l),
                U32(r) => U64(l * *r as u64),
                U64(r) => U64(l * r),
                F32(r) => F32(*l as f32 * r),
                F64(r) => F64(*l as f64 * r),
                Object(_) => U64(*l),
            },
            F32(l) => match rhs {
                None => F32(*l),
                U32(r) => F32(l * *r as f32),
                U64(r) => F32(l * *r as f32),
                F32(r) => F32(l * r),
                F64(r) => F64(*l as f64 * r),
                Object(_) => F32(*l),
            },
            F64(l) => match rhs {
                None => F64(*l),
                U32(r) => F64(l * *r as f64),
                U64(r) => F64(l * *r as f64),
                F32(r) => F64(l * *r as f64),
                F64(r) => F64(l * r),
                Object(_) => F64(*l),
            },
            Object(l) => match rhs {
                Object(r) => Object(l.clone() + r),
                _ => Object(l.clone()),
            },
        }
    }
}

impl Div<&DType> for DType {
    type Output = DType;

    fn div(self, rhs: &DType) -> Self::Output {
        use DType::*;

        match self {
            None => self,
            U32(l) => match rhs {
                None => U32(l),
                U32(r) => U32(l / r),
                U64(r) => U64(l as u64 / r),
                F32(r) => F32(l as f32 / r),
                F64(r) => F64(l as f64 / r),
                Object(_) => U32(l),
            },
            U64(l) => match rhs {
                None => U64(l),
                U32(r) => U64(l / *r as u64),
                U64(r) => U64(l / r),
                F32(r) => F32(l as f32 / r),
                F64(r) => F64(l as f64 / r),
                Object(_) => U64(l),
            },
            F32(l) => match rhs {
                None => F32(l),
                U32(r) => F32(l / *r as f32),
                U64(r) => F32(l / *r as f32),
                F32(r) => F32(l / r),
                F64(r) => F64(l as f64 / r),
                Object(_) => F32(l),
            },
            F64(l) => match rhs {
                None => F64(l),
                U32(r) => F64(l / *r as f64),
                U64(r) => F64(l / *r as f64),
                F32(r) => F64(l / *r as f64),
                F64(r) => F64(l / r),
                Object(_) => F64(l),
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
            None => None,
            U32(l) => match rhs {
                None => U32(*l),
                U32(r) => U32(l / r),
                U64(r) => U64(*l as u64 / r),
                F32(r) => F32(*l as f32 / r),
                F64(r) => F64(*l as f64 / r),
                Object(_) => U32(*l),
            },
            U64(l) => match rhs {
                None => U64(*l),
                U32(r) => U64(l / *r as u64),
                U64(r) => U64(l / r),
                F32(r) => F32(*l as f32 / r),
                F64(r) => F64(*l as f64 / r),
                Object(_) => U64(*l),
            },
            F32(l) => match rhs {
                None => F32(*l),
                U32(r) => F32(l / *r as f32),
                U64(r) => F32(l / *r as f32),
                F32(r) => F32(l / r),
                F64(r) => F64(*l as f64 / r),
                Object(_) => F32(*l),
            },
            F64(l) => match rhs {
                None => F64(*l),
                U32(r) => F64(l / *r as f64),
                U64(r) => F64(l / *r as f64),
                F32(r) => F64(l / *r as f64),
                F64(r) => F64(l / r),
                Object(_) => F64(*l),
            },
            Object(l) => Object(l.clone()),
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
