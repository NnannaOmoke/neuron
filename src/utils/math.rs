use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut2};

use super::*;
use crate::*;

pub fn forward_elimination(array: &mut ArrayViewMut2<f64>) {
    let len = array.shape()[0];
    for i in 0..len - 1 {
        for j in (i + 1)..len {
            let pivot = array[(j, i)] / array[(i, i)];
            if pivot.abs() < 1e-10f64 {
                eprintln!("Potential singular matrix encountered!");
                return;
            }
            for k in i + 1 ..=len {
                array[(j, k)] -= pivot * array[(i, k)];
            }
        }
    }
}

pub fn backward_substitution(array: &mut ArrayViewMut2<f64>) {
    let len = array.shape()[0];
    for i in (0..=(len - 1)).rev() {
        let mut sum = 0f64;
        for j in (i + 1)..len {
            sum += array[(i, j)] * array[(j, len)];
        }
        array[(i, len)] = (array[(i, len)] - sum) / array[(i, i)]
    }
}

pub fn solve_linear_systems(array: &mut ArrayViewMut2<f64>) {
    let len = array.shape()[0];
    if len == 0 {
        return;
    }
    forward_elimination(array);
    if array[(len - 1, len - 1)].abs() < 1e-10f64 {
        eprintln!("System has no unique solutions");
        return;
    }
    backward_substitution(array);
}

fn dot_d(left: ArrayView1<dtype::DType>, right: ArrayView1<dtype::DType>) -> f64 {
    assert!(left.len() == right.len());
    let mut res = dtype::DType::F64(0f64);
    zip(left.iter(), right.iter()).for_each(|(x, y)| {
        res += x * y;
    });
    res.to_f64().unwrap()
}

pub fn dot(left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
    assert!(left.len() == right.len());
    left.dot(&right)
}

pub fn sigmoid(value: f64) -> f64 {
    1f64 / (1f64 + (value.neg().exp()))
}

pub fn softmax_1d(input_vector: ArrayView1<f64>) -> Array1<f64> {
    let max = input_vector
        .map(|x| NotNan::<f64>::new(*x).unwrap())
        .iter()
        .max()
        .unwrap()
        .to_f64()
        .unwrap();
    let mut intermediate = input_vector.to_owned() - max;
    let exponented = intermediate
        .iter_mut()
        .map(|x| x.exp())
        .collect::<Array1<f64>>();
    let summed = exponented.sum();
    exponented / summed
}

pub fn one_hot_encode_1d<T>(scalar: T, size: usize) -> Array1<u32>
where
    T: ToPrimitive,
{
    let mut complete = Array1::zeros(size);
    complete[scalar.to_usize().unwrap()] = 1;
    complete
}

pub fn argmax_1d_f64(vector: ArrayView1<f64>) -> usize {
    let temp = vector.map(|x| NotNan::<f64>::new(*x).unwrap());
    let max = temp.iter().max().unwrap();
    vector
        .iter()
        .position(|x| NotNan::<f64>::new(*x).unwrap() == *max)
        .unwrap()
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::softmax_1d;

    #[test]
    fn test_softmax() {
        let array = array![3.0, 1.0, 0.2];
        dbg!(softmax_1d(array.view()));
    }
}
