mod gradient_descent;
mod newton;
mod processors;
mod util;

pub mod prelude;

#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;
use ndarray::prelude::*;

// pub trait Build {
//     type BuildResult: Step;
//     fn build(&self, n: usize) -> Self::BuildResult;
// }

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
        let transacted = util::transaction_cost(y.view(), x.view(), cost)?;
        let cash = x[0];

        let mut xr = &x * &r;
        let gross_growth = xr.scalar_sum();
        xr /= gross_growth;

        stepper.step(y.view(), x.view_mut(), r)?;

        y.assign(&xr);

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
