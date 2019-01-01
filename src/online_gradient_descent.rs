use ndarray::{Array1, ArrayView1, ArrayViewMut1};

pub struct GradientDescent {
    t: usize,
    a: f64,
    lambda: f64,
}

impl GradientDescent {
    pub fn new(a: f64, lambda: f64) -> GradientDescent {
        GradientDescent {
            t: 1,
            a: 1f64 / a,
            lambda: lambda,
        }
    }

    pub fn step(&mut self, mut x: ArrayViewMut1<f64>, r: ArrayView1<f64>) {
        let g = self.grad(x.view(), r);
        x.scaled_add(self.a / self.t as f64, &g);
        self.t += 1;
        self.project(x);
    }

    fn grad(&self, x: ArrayView1<f64>, r: ArrayView1<f64>) -> Array1<f64> {
        let xr = x.dot(&r);
        let factor = 1f64 / xr - 2f64 * self.lambda * xr.ln() / xr;
        &r * factor
    }

    fn project(&self, mut x: ArrayViewMut1<f64>) {
        let mut y = x.to_owned();
        y.as_slice_mut()
            .expect("x is not contiguous, slicing failed")
            .sort_unstable_by(|a, b| a.partial_cmp(b).expect("cannot process nans"));
        let mut s = y.sum() - 1f64;
        let mut sub = 0f64;
        let mut prev = 0f64;
        let mut nrem = y.len();
        for &yi in y.iter() {
            let diff = yi - prev;
            if s > diff * nrem as f64 {
                sub += diff;
                s -= diff * nrem as f64;
                prev = yi;
                nrem -= 1;
            } else {
                sub += s / nrem as f64;
                break;
            }
        }
        for xi in x.iter_mut() {
            *xi = 0f64.max(*xi - sub);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection() {
        let N = 10;
        let mut x = Array1::ones(N) / (N as f64);
        // let mut x_ = Array1::ones(N) / (N as f64);
        let mut gd = GradientDescent::new(1.0);
        gd.project(x.view_mut());
        assert!(x.sum() <= 1f64);
        assert!(x.sum() + (N as f64) * ::std::f64::EPSILON >= 1f64);
        assert!(x.fold(&1.0, |a, b| if a < b { a } else { b }) >= &0f64);
    }
}
