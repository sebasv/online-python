use crate::util::{project_simplex, Grad};
use crate::{Error, Reset, Step};
use ndarray::prelude::*;

pub struct GradientDescent {
    t: usize,
    a: f64,
    lambda: f64,
    cost: f64,
    grad: Grad,
}

impl GradientDescent {
    pub fn new(a: f64, lambda: f64, cost: f64, grad: Grad) -> GradientDescent {
        GradientDescent {
            t: 0,
            a,
            lambda,
            cost,
            grad,
        }
    }
}

impl Reset for GradientDescent {
    fn reset(&mut self, _n: usize) {
        self.t = 0;
    }
}

impl Step for GradientDescent {
    #[inline]
    fn step(
        &mut self,
        y: ArrayView1<f64>,
        mut x: ArrayViewMut1<f64>,
        r: ArrayView1<f64>,
    ) -> Result<(), Error> {
        self.t += 1;
        let g = self.grad.grad(y, x.view(), r, self.lambda, self.cost)?;
        x.scaled_add((self.a * (self.t as f64).sqrt()).recip(), &g);
        project_simplex(x.view_mut())?;
        Ok(())
    }
}
