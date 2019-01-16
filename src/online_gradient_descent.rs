use super::Error;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use util::{grad, project_simplex, transaction_cost};

pub struct StepResult {
    gross_growth: f64,
    transacted: f64,
    cash: f64,
}

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

    #[inline]
    pub fn step(
        &mut self,
        mut x: ArrayViewMut1<f64>,
        r: ArrayView1<f64>,
    ) -> Result<StepResult, Error> {
        let mut xr = &x * &r;
        let gross_growth = xr.scalar_sum();
        let cash = x[0];
        xr /= gross_growth;

        let g = grad(x.view(), r, self.lambda, self.cost)?;
        x.scaled_add(self.a / self.t as f64, &g);
        self.t += 1;
        project_simplex(x.view_mut())?;
        let transacted = transaction_cost(xr.view(), x.view(), self.cost)?
            .mapv(f64::abs)
            .scalar_sum();
        Ok(StepResult {
            gross_growth,
            cash,
            transacted,
        })
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
        let step_result = gd.step(x.view_mut(), r)?;
        o[0] = step_result.gross_growth;
        o[1] = step_result.cash;
        o[2] = step_result.transacted;
    }
    Ok(out)
}

pub fn step_constituents(
    a: f64,
    lambda: f64,
    cost: f64,
    r: ArrayView2<f64>,
    m: ArrayView2<bool>,
) -> Result<Array2<f64>, Error> {
    let (T, K) = m.dim();
    let mut out = Array2::ones((T, 3));
    let mut x = Array1::zeros(K);
    let mut active_set = Array1::from_elem(K, false);
    let mut active_indices = Vec::new();
    let mut gd = GradientDescent::new(a, lambda, cost);

    let starts: Vec<usize> = r
        .t()
        .outer_iter()
        .map(|col| {
            col.iter()
                .enumerate()
                .fold((true, usize::max_value()), |acc, (i, el)| {
                    if acc.0 && !el.is_nan() {
                        (false, i)
                    } else {
                        acc
                    }
                })
                .1
        })
        .collect();
    // let mut x = Array1::from_iter(starts.iter().map(|&si| if si > 0 {0f64} else {1f64}));
    // x /= x.scalar_sum();

    for (i, mi) in m.outer_iter().enumerate() {
        if active_set
            .iter()
            .zip(mi.iter())
            .fold(false, |acc, (a, mij)| acc || a != mij)
        {
            // change actives and x.
            let new_active_indices = mi
                .iter()
                .enumerate()
                .filter_map(|(i, &mij)| if mij { Some(i) } else { None })
                .collect::<Vec<usize>>();
            // create warm start for new set and
            let warm_start_k = new_active_indices.len();
            let mut y = Array1::ones(warm_start_k) / warm_start_k as f64;
            // ... do warm start ...
            let max_start =
                new_active_indices
                    .iter()
                    .fold(0, |acc, &i| if acc < starts[i] { starts[i] } else { acc });
            let warm_start_r = Array2::from_shape_fn((i - max_start, warm_start_k), |(i, j)| {
                r[(max_start + i, new_active_indices[j])]
            });

            // reset gd
            gd = GradientDescent::new(a, lambda, cost);
            for ri in warm_start_r.outer_iter() {
                gd.step(y.view_mut(), ri)?;
            }
            // volume required to alter the strategy:
            if i > 0 {
                let mut mock_x = Array1::zeros(K);
                x.iter()
                    .zip(&active_indices)
                    .for_each(|(&xi, &i)| mock_x[i] = xi);
                let mut mock_y = Array1::zeros(K);
                y.iter()
                    .zip(&new_active_indices)
                    .for_each(|(&yi, &i)| mock_y[i] = yi);
                let transacted = transaction_cost(mock_x.view(), mock_y.view(), cost)?
                    .mapv(f64::abs)
                    .scalar_sum();
                out[(i - 1, 2)] = transacted;
            }
            x = y;
            active_set = mi.to_owned();
            active_indices = new_active_indices;
        }
        // do a step, assume x fits
        let ri = Array1::from_iter(active_indices.iter().map(|&j| r[(i, j)]));
        let step_result = gd.step(x.view_mut(), ri.view())?;
        out[(i, 0)] = step_result.gross_growth;
        out[(i, 1)] = step_result.cash;
        out[(i, 2)] = step_result.transacted;
    }
    Ok(out)
}

pub fn step_constituents_fixed(
    cost: f64,
    x: ArrayView1<f64>,
    r: ArrayView2<f64>,
    m: ArrayView2<bool>,
) -> Result<Array2<f64>, Error> {
    let (T, K) = r.dim();
    let mut out = Array2::ones((T - 1, 3));
    let mut prev_invested = Array1::zeros(K);
    prev_invested[0] = 1f64;

    for ((fut_ri, mi), mut oi) in r
        .outer_iter()
        .skip(1)
        .zip(m.outer_iter())
        .zip(out.outer_iter_mut())
    {
        let active_set = mi
            .iter()
            .enumerate()
            .filter_map(|(i, &mij)| if mij { Some(i) } else { None })
            .collect::<Vec<usize>>();
        let mut y = Array1::from_iter(active_set.iter().map(|i| x[*i]));
        let active_sum = y.scalar_sum();
        y /= active_sum;

        let mut invested = Array1::zeros(K);
        for (&yi, &i) in y.iter().zip(active_set.iter()) {
            invested[i] = yi;
        }

        let fut_r = Array1::from_iter(active_set.iter().map(|i| fut_ri[*i]));
        let transacted = transaction_cost(prev_invested.view(), invested.view(), cost)?
            .mapv(f64::abs)
            .scalar_sum();

        oi[0] = y.dot(&fut_r) * (1f64 - cost * transacted);
        oi[1] = y[0];
        oi[2] = transacted;
        prev_invested = &invested * &fut_ri.mapv(|rij| if rij.is_nan() { 0f64 } else { rij });
        prev_invested /= prev_invested.scalar_sum();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use rand::distributions::LogNormal;
    use rand::prelude::*;
    // use ndarray::prelude::*;
    use super::*;
    #[test]
    fn test_step_constituents() {
        let (T, K) = (700, 100);
        // let x0 = Array1::ones(K) / K as f64;
        let mut x0 = Array1::zeros(K);
        x0[0] = 1f64;
        let ln = LogNormal::new(1f64, 0.5);
        let r = Array2::from_shape_fn((T, K), |(_, j)| {
            if j == 0 {
                1f64
            } else {
                ln.sample(&mut thread_rng())
            }
        });
        let m = Array2::from_shape_fn((T, K), |(i, j)| if j == 0 { true } else { random() });
        // let m = Array2::from_elem((T,K), true);
        let a = step_constituents(1f64, 0f64, 0.002, r.view(), m.view()).unwrap();
        // println!("{}", a);
        // println!("{}", r);
        // println!("{}", m);
    }
}
