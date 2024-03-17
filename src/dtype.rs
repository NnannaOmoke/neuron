use std::fmt::Display;

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

impl DType {
    pub fn cast(&self, rhs: &String) -> Self {
        match self {
            DType::None => match rhs.to_lowercase().as_str() {
                _ => {
                    eprintln!("None type is not convertible.");
                    DType::None
                }
            },
            DType::F32(value) => match rhs.to_lowercase().as_str() {
                "none" => DType::None,
                "u32" => DType::U32(*value as u32),
                "u64" => DType::U64(*value as u64),
                "f32" => DType::F32(*value as f32),
                "f64" => DType::F64(*value as f64),
                "string" => DType::Object(value.to_string()),
                _ => panic!("Invalid datatype variant given"),
            },
            DType::F64(value) => match rhs.to_lowercase().as_str() {
                "none" => DType::None,
                "u32" => DType::U32(*value as u32),
                "u64" => DType::U64(*value as u64),
                "f32" => DType::F32(*value as f32),
                "f64" => DType::F64(*value as f64),
                "string" => DType::Object(value.to_string()),
                _ => panic!("Invalid datatype variant given"),
            },
            DType::U32(value) => match rhs.to_lowercase().as_str() {
                "none" => DType::None,
                "u32" => DType::U32(*value as u32),
                "u64" => DType::U64(*value as u64),
                "f32" => DType::F32(*value as f32),
                "f64" => DType::F64(*value as f64),
                "string" => DType::Object(value.to_string()),
                _ => panic!("Invalid datatype variant given"),
            },
            DType::U64(value) => match rhs.to_lowercase().as_str() {
                "none" => DType::None,
                "u32" => DType::U32(*value as u32),
                "u64" => DType::U64(*value as u64),
                "f32" => DType::F32(*value as f32),
                "f64" => DType::F64(*value as f64),
                "string" => DType::Object(value.to_string()),
                _ => panic!("Invalid datatype variant given"),
            },
            DType::Object(value) => match rhs.to_lowercase().as_str() {
                "none" => DType::None,
                "u32" => DType::U32(str::parse::<u32>(&value).expect("Parse failed!")),
                "u64" => DType::U64(str::parse::<u64>(&value).expect("Parse failed!")),
                "f32" => DType::F32(str::parse::<f32>(&value).expect("Parse failed!")),
                "f64" => DType::F64(str::parse::<f64>(&value).expect("Parse failed!")),
                "string" => DType::Object(value.to_string()),
                _ => panic!("Invalid datatype variant given"),
            },
        }
    }

    pub fn display_type(&self) -> String {
        let mut string = String::new();
        match self {
            DType::None => string.push_str("None"),
            DType::F32(_) => string.push_str("f32"),
            DType::F64(_) => string.push_str("f64"),
            DType::U32(_) => string.push_str("u32"),
            DType::U64(_) => string.push_str("u64"),
            DType::Object(_) => string.push_str("object"),
        }
        string
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
