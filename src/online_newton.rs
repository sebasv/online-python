use super::Error;
use ndarray::{Array2, ArrayView1, ArrayViewMut1};
use util::{grad, project_simplex_general};

pub struct Newton {
    #[allow(non_snake_case)]
    approx_hessian_inv: Array2<f64>,
    #[allow(non_snake_case)]
    approx_hessian: Array2<f64>,
    max_iter: usize,
    beta: f64,
    lambda: f64,
    cost: f64,
}

impl Newton {
    pub fn new(beta: f64, eps: f64, n: usize, max_iter: usize, lambda: f64, cost: f64) -> Newton {
        Newton {
            approx_hessian: Array2::eye(n) * eps,
            approx_hessian_inv: Array2::eye(n) / eps,
            max_iter,
            beta,
            lambda,
            cost,
        }
    }

    pub fn step(&mut self, mut x: ArrayViewMut1<f64>, r: ArrayView1<f64>) -> Result<(), Error> {
        let g = grad(x.view(), r, self.lambda, self.cost)?;
        self.update_approx_hessian(g.view());
        x -= &(self.approx_hessian_inv.dot(&g) / self.beta);
        let projected =
            project_simplex_general(x.view(), self.approx_hessian.view(), self.max_iter)?;
        x.assign(&projected);
        Ok(())
    }

    #[allow(non_snake_case)]
    fn update_approx_hessian(&mut self, g: ArrayView1<f64>) {
        let v = g
            .into_shape((g.len(), 1))
            .expect("g reshape, should not fail");

        self.approx_hessian += &(v.dot(&v.t()));

        let u = self.approx_hessian_inv.dot(&v);
        debug_assert!(u.shape() == v.shape());
        self.approx_hessian_inv -= &(u.dot(&u.t()) / (1f64 + u.dot(&g)));
    }
}
