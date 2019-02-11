mod gradient_descent;
mod newton;
mod processors;
mod util;

pub mod prelude;

#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;
use ndarray::prelude::*;

pub trait Build {
    type BuildResult: Step;
    fn build(&self, n: usize) -> Self::BuildResult;
}

pub trait Step {
    fn step(&mut self, x: ArrayViewMut1<f64>, r: ArrayView1<f64>) -> Result<(), Error>;
}

pub struct StepResult {
    gross_growth: f64,
    transacted: f64,
    cash: f64,
}

impl StepResult {
    pub fn step<S>(
        mut x: ArrayViewMut1<f64>,
        r: ArrayView1<f64>,
        cost: f64,
        stepper: &mut S,
    ) -> Result<StepResult, Error>
    where
        S: Step,
    {
        let mut xr = &x * &r;
        let gross_growth = xr.scalar_sum();
        let cash = x[0];
        xr /= gross_growth;

        stepper.step(x.view_mut(), r)?;

        let transacted = util::transaction_cost(xr.view(), x.view(), cost)?;
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
