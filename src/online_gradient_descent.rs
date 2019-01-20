use ndarray::prelude::*;
use util::{grad, project_simplex};
use {Build, Error, Step};

pub struct GradientBuilder {
    a: f64,
    lambda: f64,
    cost: f64,
}

impl GradientBuilder {
    pub fn new(a: f64, lambda: f64, cost: f64) -> GradientBuilder {
        GradientBuilder { a, lambda, cost }
    }
}

impl Build for GradientBuilder {
    type BuildResult = GradientDescent;
    fn build(&self, _n: usize) -> GradientDescent {
        GradientDescent::new(self.a, self.lambda, self.cost)
    }
}

#[derive(Clone)]
pub struct GradientDescent {
    t: usize,
    a: f64,
    lambda: f64,
    cost: f64,
}

impl GradientDescent {
    pub fn new(a: f64, lambda: f64, cost: f64) -> GradientDescent {
        GradientDescent {
            t: 1,
            a: 1f64 / a,
            lambda,
            cost,
        }
    }
}

impl Step for GradientDescent {
    #[inline]
    fn step(&mut self, mut x: ArrayViewMut1<f64>, r: ArrayView1<f64>) -> Result<(), Error> {
        let g = grad(x.view(), r, self.lambda, self.cost)?;
        x.scaled_add(self.a / self.t as f64, &g);
        self.t += 1;
        project_simplex(x.view_mut())?;
        Ok(())
    }
}
