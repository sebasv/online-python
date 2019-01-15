use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use util::{grad, project_simplex, transaction_cost};
use super::Error;

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

    pub fn step(&mut self, mut x: ArrayViewMut1<f64>, r: ArrayView1<f64>) -> Result<(), Error>{

        let g = grad(x.view(), r, self.lambda, self.cost)?;
        x.scaled_add(self.a / self.t as f64, &g);
        self.t += 1;
        project_simplex(x)?;
        Ok(())
    }
}

pub fn step_all(
    a: f64,
    lambda: f64,
    cost: f64,
    x0: ArrayView1<f64>,
    data: ArrayView2<f64>,
) -> Result<Array2<f64>, Error> {
    let mut gd = GradientDescent::new(a, lambda, cost);
    let mut x = x0.to_owned();
    let mut out = Array2::zeros((data.shape()[0], 3));
    for (r, mut o) in data.outer_iter().zip(out.outer_iter_mut()) {
        o[0] = x.dot(&r);
        o[1] = x[0];
        let prev = &x * &r;
        gd.step(x.view_mut(), r)?;
        o[2] = (&prev - &x).mapv(f64::abs).scalar_sum();
    }
    Ok(out)
}

pub fn step_constituents(
    a: f64,
    lambda: f64,
    cost: f64,
    x0: ArrayView1<f64>,
    r: ArrayView2<f64>,
    m: ArrayView2<bool>,
) -> Result<Array2<f64>, Error> {
    let mut gd = GradientDescent::new(a, lambda, cost);

    let (T, K) = r.dim();
    let mut out = Array2::ones((T - 1, 3));
    // let mut x = x0.to_owned();
    let mut prev_invested = Array1::zeros(K);
    prev_invested[0] = 1f64;
    let mut invested = x0.to_owned();

    for (((ri, mi), mut oi), fut_ri) in r
        .outer_iter()
        .zip(m.outer_iter())
        .zip(out.outer_iter_mut())
        .zip(r.outer_iter().skip(1))
    {
        let active_set = mi
            .iter()
            .enumerate()
            .filter_map(|(i, &mij)| if mij { Some(i) } else { None })
            .collect::<Vec<usize>>();
        let mut y = Array1::from_iter(active_set.iter().map(|i| /*x*/invested[*i]));
        let active_sum = y.scalar_sum();
        y /= active_sum;
        let z = Array1::from_iter(active_set.iter().map(|i| ri[*i]));

        gd.step(y.view_mut(), z.view())?;

        /*let mut */invested = Array1::zeros(K);
        for (&yi, &i) in y.iter().zip(active_set.iter()) {
            invested[i] = yi;
        }

        let fut_r = Array1::from_iter(active_set.iter().map(|i| fut_ri[*i]));
        let transacted = transaction_cost(prev_invested.view(), invested.view(), cost)?
            .mapv(f64::abs)
            .scalar_sum();
        oi[0] = y.dot(&fut_r)*(1f64 - cost*transacted);
        oi[1] = y[0];
        oi[2] = transacted;
        prev_invested = &invested * &fut_ri.mapv(|rij| if rij.is_nan() {0f64} else {rij});
        prev_invested /= prev_invested.scalar_sum();

        // // This does not cover the case that `inactive_sum == 0`.
        // let mut y_iter = y.iter();
        // let inactive_sum = x.scalar_sum() - active_sum;
        // let w1 = y.len() as f64 / K as f64;
        // let w2 = (1f64 - w1) / inactive_sum;
        // for (&mij, mut xj) in mi.iter().zip(x.iter_mut()) {
        //     *xj = if mij {
        //         y_iter.next().unwrap() * w1
        //     } else {
        //         *xj * w2
        //     };
        // }
    }
    Ok(out)
}


pub fn step_constituents_fixed(
    a: f64,
    lambda: f64,
    cost: f64,
    x: ArrayView1<f64>,
    r: ArrayView2<f64>,
    m: ArrayView2<bool>,
) -> Result<Array2<f64>, Error> {
    let (T, K) = r.dim();
    let mut out = Array2::ones((T - 1, 3));
    let mut prev_invested = Array1::zeros(K);
    prev_invested[0] = 1f64;

    for (((ri, mi), mut oi), fut_ri) in r
        .outer_iter()
        .zip(m.outer_iter())
        .zip(out.outer_iter_mut())
        .zip(r.outer_iter().skip(1))
    {
        let active_set = mi
            .iter()
            .enumerate()
            .filter_map(|(i, &mij)| if mij { Some(i) } else { None })
            .collect::<Vec<usize>>();
        let mut y = Array1::from_iter(active_set.iter().map(|i| x[*i]));
        let active_sum = y.scalar_sum();
        y /= active_sum;
        let z = Array1::from_iter(active_set.iter().map(|i| ri[*i]));

        let mut invested = Array1::zeros(K);
        for (&yi, &i) in y.iter().zip(active_set.iter()) {
            invested[i] = yi;
        }

        let fut_r = Array1::from_iter(active_set.iter().map(|i| fut_ri[*i]));
        let transacted = transaction_cost(prev_invested.view(), invested.view(), cost)?
            .mapv(f64::abs)
            .scalar_sum();
        oi[0] = y.dot(&fut_r)*(1f64 - cost*transacted);
        oi[1] = y[0];
        oi[2] = transacted;
        prev_invested = &invested * &fut_ri.mapv(|rij| if rij.is_nan() {0f64} else {rij});
        prev_invested /= prev_invested.scalar_sum();

    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand::distributions::LogNormal;
    // use ndarray::prelude::*;
    use super::*;
    #[test]
    fn test_step_constituents() {
        let (T, K) = (5, 3);
        // let x0 = Array1::ones(K) / K as f64;
        let mut x0 = Array1::zeros(K);
        x0[0] = 1f64;
        let ln = LogNormal::new(1f64, 0.5);
        let r = Array2::from_shape_fn((T,K), |(i,j)| if j==0 {1f64} else {ln.sample(&mut thread_rng())});
        let m = Array2::from_shape_fn((T,K), |(i,j)| if j==0 {true} else {random()});
        // let m = Array2::from_elem((T,K), true);
        let a = step_constituents(1f64, 0f64, 0.002, x0.view(), r.view(), m.view()).unwrap();
        println!("{}", a);
        println!("{}", r);
        println!("{}", m);
    }
}