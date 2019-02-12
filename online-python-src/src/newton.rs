use crate::util::{project_simplex_general, Grad};
use crate::{Error, Reset, Step};
use ndarray::prelude::*;

// pub struct NewtonBuilder {
//     beta: f64,
//     max_iter: usize,
//     lambda: f64,
//     cost: f64,
//     grad: Grad,
// }

// impl NewtonBuilder {
//     pub fn new(beta: f64, max_iter: usize, lambda: f64, cost: f64, grad: Grad) -> NewtonBuilder {
//         NewtonBuilder {
//             beta,
//             max_iter,
//             lambda,
//             cost,
//             grad,
//         }
//     }
// }

// impl Build for NewtonBuilder {
//     type BuildResult = Newton;
//     fn build(&self, n: usize) -> Newton {
//         Newton::new(
//             self.beta,
//             self.max_iter,
//             self.lambda,
//             self.cost,
//             self.grad,
//             n,
//         )
//     }
// }

pub struct Newton {
    approx_hessian_inv: Array2<f64>,
    pub approx_hessian: Array2<f64>,
    max_iter: usize,
    t: usize,
    beta: f64,
    lambda: f64,
    cost: f64,
    grad: Grad,
}

impl Newton {
    pub fn new(beta: f64, max_iter: usize, lambda: f64, cost: f64, grad: Grad, n: usize) -> Newton {
        let eps = beta.powi(2).recip();
        Newton {
            approx_hessian: Array2::eye(n) * eps,
            approx_hessian_inv: Array2::eye(n) / eps,
            max_iter,
            t: 0,
            beta,
            lambda,
            cost,
            grad,
        }
    }

    fn update_approx_hessian(&mut self, g: ArrayView1<f64>) {
        let v = g
            .into_shape((g.len(), 1))
            .expect("g reshape, should not fail");

        self.approx_hessian += &(v.dot(&v.t()));

        let u = self.approx_hessian_inv.dot(&v);
        self.approx_hessian_inv -=
            &(u.dot(&u.t()) / (1f64 + u.into_shape(g.len()).unwrap().dot(&g)));
    }
}

impl Reset for Newton {
    fn reset(&mut self, n: usize) {
        self.t = 0;
        let eps = self.beta.powi(2).recip();
        self.approx_hessian = Array2::eye(n) * eps;
        self.approx_hessian_inv = Array2::eye(n) / eps;
    }
}

impl Step for Newton {
    #[inline]
    fn step(
        &mut self,
        y: ArrayView1<f64>,
        mut x: ArrayViewMut1<f64>,
        r: ArrayView1<f64>,
    ) -> Result<(), Error> {
        self.t += 1;
        let g = self.grad.grad(y, x.view(), r, self.lambda, self.cost)?;
        // let g = grad(x.view(), r, self.lambda, self.cost)?;
        self.update_approx_hessian(g.view());
        x -= &(self.approx_hessian_inv.dot(&g) / self.beta);

        // divide the appprox-hessian by t. Does not theoretically alter the results,
        // but solves a numerical issue where the elements of approx_hessian grow too large
        let projected = project_simplex_general(
            x.view(),
            (&self.approx_hessian / self.t as f64).view(),
            self.max_iter,
        )?;
        x.assign(&projected);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newton() {
        let mut newt = Newton::new(1e-3, 100, 0f64, 0f64, 4);
        let mut x = arr1(&[0.25, 0.25, 0.25, 0.25]);
        let r = arr1(&[1., 1.1, 0.9, 1.2]);
        newt.step(x.view_mut(), r.view()).unwrap();
        println!("{:?}", x);
    }

    #[test]
    fn test_outer() {
        let a = arr2(&[[1.], [2.]]);
        let v = a.dot(&a.t());
        println!("{:?}", v);
        assert!(v.dim() == (2, 2));
        assert!(v.all_close(&arr2(&[[1., 2.], [2., 4.]]), 1e-9));
    }
}
