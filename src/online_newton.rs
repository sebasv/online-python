use ndarray::{Array2, ArrayView1, ArrayViewMut1};
use util::grad;
use super::Error;

pub struct Newton {
    #[allow(non_snake_case)]
    Ai: Array2<f64>,
    #[allow(non_snake_case)]
    A: Array2<f64>,
    beta: f64,
    lambda: f64,
    cost: f64,
}

impl Newton {
    pub fn new(beta: f64, eps: f64, n: usize, lambda: f64, cost: f64) -> Newton {
        Newton {
            A: Array2::eye(n) * eps,
            Ai: Array2::eye(n) / eps,
            beta,
            lambda,
            cost,
        }
    }

    pub fn step(&mut self, mut x: ArrayViewMut1<f64>, r: ArrayView1<f64>) -> Result<(), Error> {
        let g = grad(x.view(), r, self.lambda, self.cost)?;
        self.update_A(g.view());
        x -= &(self.Ai.dot(&g) / self.beta);
        self.project(x);
        Ok(())
    }

    fn project(&self, mut x: ArrayViewMut1<f64>) {
        // TODO solve how to efficiently minimize_y (x-y)'self.A(x-y)
    }

    #[allow(non_snake_case)]
    fn update_A(&mut self, g: ArrayView1<f64>) {
        let v = g
            .into_shape((g.len(), 1))
            .expect("g reshape, should not fail");

        self.A += &(v.dot(&v.t()));

        let u = self.Ai.dot(&v);
        debug_assert!(u.shape() == v.shape());
        self.Ai -= &(u.dot(&u.t()) / (1f64 + u.dot(&g)));
    }
}
