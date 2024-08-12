#![allow(unused_variables)]
use std::default;

//temporary thing for now
use super::*;
use crate::{linear_models::LinearRegressorBuilder, *};
use ndarray::{ArrayView1, ArrayView2};
use num_traits::Pow;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Default)]
pub enum Average {
    #[default]
    Macro,
    Weighted,
    None,
}

impl Average {
    fn scoring(&self, input: &[(f64, f64)]) -> Vec<f64> {
        match self {
            Average::Macro => {
                vec![
                    input
                        .iter()
                        .fold(0f64, |accum, &(values, _)| accum + values)
                        / input.len() as f64,
                ]
            }
            Average::Weighted => {
                let value = input
                    .iter()
                    .fold(0f64, |accum, &(value, support)| accum + (support * value));
                let length: f64 = input.iter().map(|(_, support)| support).sum();
                let value = value / length;
                vec![value]
            }
            Average::None => input.iter().map(|&(value, _)| value).collect(),
        }
    }
}

pub fn mean_abs_error(target: ArrayView1<f64>, predicted: ArrayView1<f64>) -> Vec<f64> {
    assert!(target.len() == predicted.len());
    let len = target.len() as f64;
    let mut total = 0f64;
    zip(target, predicted).for_each(|(x, y)| total += (x - y).abs());
    vec![total / len]
}

pub fn mean_squared_error(target: ArrayView1<f64>, predicted: ArrayView1<f64>) -> Vec<f64> {
    assert!(target.len() == predicted.len());
    let len = target.len() as f64;
    let mut total = 0f64;
    zip(target, predicted).for_each(|(x, y)| total += (x - y).pow(2));
    vec![total / len]
}

pub fn root_mean_square_error(target: ArrayView1<f64>, predicted: ArrayView1<f64>) -> Vec<f64> {
    vec![mean_squared_error(target, predicted)[0].sqrt()]
}

pub fn r2_score(target: ArrayView1<f64>, predicted: ArrayView1<f64>) -> Vec<f64> {
    assert!(target.len() == predicted.len());
    let num = zip(target, predicted).fold(0f64, |accum, (&y, &yhat)| accum + (y - yhat).pow(2));
    let ybar = target.mean().unwrap();
    let denum = target.map(|y| (y - ybar).pow(2)).sum();
    vec![1f64 - (num / denum)]
}

pub fn mean_squared_log_error(target: ArrayView1<f64>, predicted: ArrayView1<f64>) -> Vec<f64> {
    assert!(target.len() == predicted.len());
    vec![
        zip(target, predicted).fold(0f64, |accum, (&y, &yhat)| {
            accum + ((1f64 + y).ln() - (1f64 + yhat).ln()).pow(2)
        }) / target.len() as f64,
    ]
}

pub fn accuracy(target: ArrayView1<u32>, predicted: ArrayView1<u32>) -> Vec<f64> {
    assert!(target.len() == predicted.len());
    let correct = zip(target, predicted).filter(|(x, y)| x == y).count();
    vec![correct as f64 / target.len() as f64]
}

pub fn build_confusion_matrix(target: ArrayView1<u32>, predicted: ArrayView1<u32>) -> Array2<u32> {
    assert!(target.len() == predicted.len());
    let setted: HashSet<&u32> = HashSet::from_iter(target.iter());
    let nunique = setted.len();
    let mut square = Array2::zeros((nunique, nunique));
    zip(target, predicted).for_each(|(&t, &p)| {
        square[(t as usize, p as usize)] += 1;
    });
    square
}

pub fn precision(
    target: ArrayView1<u32>,
    predicted: ArrayView1<u32>,
    average: Average,
) -> Vec<f64> {
    //build confusion matrix and evaluate based on average
    let confusion = build_confusion_matrix(target, predicted);
    let complete = confusion
        .columns()
        .into_iter()
        .enumerate()
        .map(|(index, column)| {
            let tp = confusion[(index, index)];
            let fp: u32 = column
                .iter()
                .enumerate()
                .filter(|&(f, _)| f != index)
                .map(|(_, val)| val)
                .sum();
            let all = tp as f64 + fp as f64;
            let pres = tp as f64 / (tp as f64 + fp as f64);
            (pres, all)
        })
        .collect::<Vec<(f64, f64)>>();
    let pres = average.scoring(&complete);
    pres
}

pub fn recall(target: ArrayView1<u32>, predicted: ArrayView1<u32>, average: Average) -> Vec<f64> {
    let confusion = build_confusion_matrix(target, predicted);
    let complete = confusion
        .rows()
        .into_iter()
        .enumerate()
        .map(|(index, row)| {
            let tp = confusion[(index, index)];
            let false_neg: u32 = row
                .iter()
                .enumerate()
                .filter(|&(f, _)| f != index)
                .map(|(_, val)| val)
                .sum();
            let all = tp as f64 + false_neg as f64;
            let rec = tp as f64 / all;
            (rec, all)
        })
        .collect::<Vec<(f64, f64)>>();
    let rec = average.scoring(&complete);
    rec
}
