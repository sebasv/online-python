use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use util::project_simplex;

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
        project_simplex(x);
    }

    fn grad(&self, x: ArrayView1<f64>, r: ArrayView1<f64>) -> Array1<f64> {
        let xr = x.dot(&r);
        let factor = 1f64 / xr - 2f64 * self.lambda * xr.ln().min(0f64) / xr;
        &r * factor
    }
}

pub fn step_all(a: f64, lambda: f64, x0: ArrayView1<f64>, data: ArrayView2<f64>) -> Array2<f64> {
    let mut gd = GradientDescent::new(a, lambda);
    let mut x = x0.to_owned();
    let mut out = Array2::zeros((data.shape()[0], 3));
    for (r, mut o) in data.outer_iter().zip(out.outer_iter_mut()) {
        o[0] = x.dot(&r);
        o[1] = x[0];
        let prev = &x * &r;
        gd.step(x.view_mut(), r);
        o[2] = (&prev - &x).mapv(f64::abs).scalar_sum();
    }
    out
}

pub fn step_constituents(
    a: f64,
    lambda: f64,
    x0: ArrayView1<f64>,
    r: ArrayView2<f64>,
    m: ArrayView2<bool>,
) -> Array1<f64> {
    let mut gd = GradientDescent::new(a, lambda);

    let (T, K) = r.dim();
    let mut out = Array1::ones(T - 1);
    let mut x = x0.to_owned();
    // let mut n = Array1::from_elem(K, false);
    for (((ri, mi), mut oi), fut_ri) in r
        .outer_iter()
        .zip(m.outer_iter())
        .zip(out.iter_mut())
        .zip(r.outer_iter().skip(1))
    {
        let active_set = mi
            .iter()
            .enumerate()
            .filter_map(|(i, &mij)| if mij { Some(i) } else { None })
            .collect::<Vec<usize>>();
        let mut y = Array1::from_iter(active_set.iter().map(|i| x[*i]));
        y /= y.scalar_sum();
        let z = Array1::from_iter(active_set.iter().map(|i| ri[*i]));

        gd.step(y.view_mut(), z.view());

        *oi = y.dot(&Array1::from_iter(active_set.iter().map(|i| fut_ri[*i])));

        let mut y_iter = y.iter();
        let w1 = y.len() as f64 / K as f64;
        let w2 = 1f64 - w1; // (1f64 - w1) / x.scalar_sum();
        for (&mij, mut xj) in mi.iter().zip(x.iter_mut()) {
            *xj = if mij {
                y_iter.next().unwrap() * w1
            } else {
                *xj * w2
            };
        }
    }
    out
}
