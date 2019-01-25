pub mod online_gradient_descent;
pub mod online_newton;
pub mod processors;
pub mod util;

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
    fn step<S>(
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

#[derive(Debug)]
pub enum Error {
    NaNError(&'static str),
    ContiguityError(&'static str),
    SolveError(&'static str),
    InvalidMethodError(&'static str),
    ConvergenceError(&'static str),
}
