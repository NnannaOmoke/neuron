use super::*;
use crate::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, ArrayViewMut2};
use rand::prelude::ThreadRng;
use rand::seq::SliceRandom;
use std::cmp;
use std::collections::HashSet;
use std::hash::Hash;

pub fn forward_elimination(array: &mut ArrayViewMut2<f64>) {
    let len = array.shape()[0];
    for i in 0..len - 1 {
        for j in (i + 1)..len {
            let pivot = array[(j, i)] / array[(i, i)];
            if pivot.abs() < 1e-10f64 {
                eprintln!("Potential singular matrix encountered!");
                return;
            }
            for k in i + 1..=len {
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
    let mut max = 0;
    for (index, elem) in vector.iter().enumerate() {
        if f64::max(*elem, vector[max]) == *elem {
            max = index;
        }
    }
    max
}

pub fn argmax_1d<T: PartialOrd>(vector: ArrayView1<T>) -> usize {
    let mut max = 0;
    for (index, elem) in vector.iter().enumerate() {
        if elem >= &vector[max] {
            max = index;
        }
    }
    max
}

pub fn argmin_1d<T: PartialOrd>(vector: ArrayView1<T>) -> usize {
    let mut min = 0;
    for (index, elem) in vector.iter().enumerate() {
        if elem <= &vector[min] {
            min = index;
        }
    }
    min
}

pub fn argmin_1d_f64(vector: ArrayView1<f64>) -> usize {
    let mut min = 0;
    for (index, elem) in vector.iter().enumerate() {
        if f64::min(*elem, vector[min]) == *elem {
            min = index;
        }
    }
    min
}

pub fn shuffle_1d<T: Clone>(vector: &mut Array1<T>) {
    let mut holder = vector.clone().to_vec();
    let mut rngs = rand::thread_rng();
    holder.shuffle(&mut rngs);
    *vector = Array1::from_vec(holder);
}

pub(crate) fn into_row_matrix<T: Clone>(vector: ArrayView1<T>) -> Array2<T> {
    Array2::from_shape_fn((vector.len(), 1), |(x, _)| vector[x].clone())
}

pub(crate) fn into_column_matrix<T: Clone>(vector: ArrayView1<T>) -> Array2<T> {
    Array2::from_shape_fn((1, vector.len()), |(_, y)| vector[y].clone())
}

pub fn outer_product<T: 'static + Float>(
    input_one: ArrayView1<T>,
    input_two: ArrayView1<T>,
) -> Array2<T> {
    into_row_matrix(input_one).dot(&into_column_matrix(input_two).view())
}

pub fn nunique<T: Clone + Hash + Eq>(vector: ArrayView1<T>) -> usize {
    let set: HashSet<&T> = HashSet::from_iter(vector.iter());
    set.len()
}

pub fn nunique_f64(vector: ArrayView1<f64>) -> usize {
    let set: HashSet<NotNan<f64>> =
        HashSet::from_iter(vector.iter().map(|&f| NotNan::new(f).unwrap()));
    set.len()
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_softmax() {
        let array = array![3.0, 1.0, 0.2];
        dbg!(softmax_1d(array.view()));
    }

    #[test]
    fn test_outer_mul() {
        let one = array![0, 1, 2, 3, 4].map(|x| *x as f32);
        let two = array![5, 6, 7, 8, 9].map(|x| *x as f32);
        dbg!(outer_product(one.view(), two.view()));
    }
}
