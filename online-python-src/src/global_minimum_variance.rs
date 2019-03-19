use crate::util::project_simplex_general;
use crate::{Error, Reset, Step};
use ndarray::prelude::*;

pub struct GMV {
    cov_inv: Array2<f64>,
    mean: Array1<f64>,
    max_iter: usize,
    eps: f64,
    positive: bool,
    t: usize,
}

impl GMV {
    pub fn new(eps: f64, max_iter: usize, n: usize, positive: bool) -> GMV {
        GMV {
            cov_inv: Array2::eye(n) / eps,
            mean: Array1::zeros(n),
            max_iter,
            t: 0,
            eps,
            positive,
        }
    }

    fn sherman_morrison(&mut self, u: ArrayView1<f64>, v: ArrayView1<f64>) {
        let qu = self.cov_inv.dot(&u);
        let qv = self.cov_inv.dot(&v);
        let uqv = qu.dot(&v);
        let qu = qu.insert_axis(Axis(1));
        let qv = qv.insert_axis(Axis(0));
        self.cov_inv -= &(qu.dot(&qv) / (1f64 + uqv));
    }
}

impl Reset for GMV {
    fn reset(&mut self, n: usize) {
        self.t = 0;
        self.cov_inv = Array2::eye(n) / self.eps;
        self.mean = Array1::zeros(n);
    }
}

impl Step for GMV {
    #[inline]
    fn step(
        &mut self,
        _y: ArrayView1<f64>,
        mut x: ArrayViewMut1<f64>,
        r: ArrayView1<f64>,
    ) -> Result<(), Error> {
        // application of the Welford online algorithm for (T-1)*variance.
        // But we don't care about scale.
        self.t += 1;
        let dif = &r - &self.mean;
        self.mean
            .scaled_add((self.t as f64).recip(), &(&r - &self.mean));
        let dif_new = &r - &self.mean;
        self.sherman_morrison(dif.view(), dif_new.view());

        // divide the appprox-hessian by t. Does not theoretically alter the results,
        // but solves a numerical issue where the elements of approx_hessian grow too large
        let projected = if self.positive {
            project_simplex_general(
                Array1::zeros(self.cov_inv.dim().0).view(),
                self.cov_inv.view(),
                self.max_iter,
            )?
        } else {
            let t = self.cov_inv.dot(&Array1::ones(self.cov_inv.dim().0));
            &t / t.scalar_sum()
        };
        x.assign(&projected);
        Ok(())
    }
}
