mod gradient_descent;
mod newton;
mod processors;
mod util;

pub mod prelude;

#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;
use ndarray::prelude::*;

use std::f64;

pub trait Reset {
    fn reset(&mut self, n: usize);
}

pub trait Step {
    fn step(
        &mut self,
        y: ArrayView1<f64>,
        x: ArrayViewMut1<f64>,
        r: ArrayView1<f64>,
    ) -> Result<(), Error>;
}

pub struct StepResult {
    pub gross_growth: f64,
    pub transacted: f64,
    pub cash: f64,
}

fn validate_input(x: ArrayView1<f64>, y: ArrayView1<f64>) -> Result<(), Error> {
    let x_sum = x.scalar_sum() - 1f64;
    if x_sum.abs() > x.len().max(100) as f64 * f64::EPSILON {
        panic!("invalid allocation vector during step: sum(x)-1={}", x_sum);
    }
    let x_min = x.fold(1f64, |acc, &e| acc.min(e));
    if x_min < 0f64 * f64::EPSILON {
        panic!("invalid allocation vector during step: min(x)={}", x_min);
    }
    let y_sum = y.scalar_sum() - 1f64;
    if y_sum.abs() > y.len().max(100) as f64 * f64::EPSILON {
        panic!("invalid allocation vector during step: sum(y)-1={}", y_sum);
    }
    let y_min = y.fold(1f64, |acc, &e| acc.min(e));
    if y_min < 00f64 * f64::EPSILON {
        panic!("invalid allocation vector during step: min(y)={}", y_min);
    }
    Ok(())
}

impl StepResult {
    pub fn step<S>(
        mut y: ArrayViewMut1<f64>,
        mut x: ArrayViewMut1<f64>,
        r: ArrayView1<f64>,
        cost: f64,
        stepper: &mut S,
    ) -> Result<StepResult, Error>
    where
        S: Step,
    {
        validate_input(x.view(), y.view())?;
        // y tracks the actual allocation, x the desired one.
        let transacted = util::transaction_cost(y.view(), x.view(), cost)?;
        let cash = x[0];

        y.assign(&(&x * &r));
        let gross_growth = y.scalar_sum();
        y /= gross_growth;

        stepper.step(y.view(), x.view_mut(), r)?;

        Ok(StepResult {
            gross_growth,
            cash,
            transacted,
        })
    }
}

impl std::convert::From<StepResult> for Array1<f64> {
    fn from(sr: StepResult) -> Array1<f64> {
        let mut vec = Vec::with_capacity(3);
        vec.push(sr.gross_growth);
        vec.push(sr.transacted);
        vec.push(sr.cash);
        Array1::from_vec(vec)
    }
}

#[derive(Debug)]
pub struct Error {
    pub message: String,
}

impl Error {
    pub fn new(message: &str) -> Error {
        Error {
            message: message.to_owned(),
        }
    }
}
