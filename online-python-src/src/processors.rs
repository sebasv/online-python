use crate::util::transaction_cost;
use crate::{Error, Reset, Step, StepResult};
use ndarray::prelude::*;
use std::f64;

pub fn step_constituents<M>(
    cost: f64,
    r: ArrayView2<f64>,
    m: ArrayView2<bool>,
    mut method: M,
) -> Result<Array2<f64>, Error>
where
    M: Reset + Step,
{
    let (n_obs, n_assets) = m.dim();
    if r.dim().0 != n_obs || r.dim().1 != n_assets {
        return Err(Error::new("dimensions of r and m must agree"));
    }

    let mut out = Array2::zeros((n_obs, 4 + n_assets));
    let mut reset_count = 0f64;
    let mut x = Array1::ones(n_assets);
    x /= n_assets as f64;
    let mut y = Array1::ones(n_assets);
    y /= n_assets as f64;
    let mut active_set = Array1::from_elem(n_assets, true);
    let mut active_indices: Vec<usize> = (0..n_assets).collect();

    for (i, mi) in m.outer_iter().enumerate() {
        let manually_transacted = if (&active_set ^ &mi).iter().any(|el| *el) {
            reset_count += 1f64;
            let mut s = Array1::zeros(mi.len());
            let mut t = Array1::zeros(mi.len());
            active_indices
                .iter()
                .zip(x.iter())
                .zip(y.iter())
                .for_each(|((&j, &xj), &yj)| {
                    s[j] = xj;
                    t[j] = yj;
                });
            let mut total_sold = 0f64;
            let mut x_transferred = 0f64;
            let mut new_indices = Vec::new();
            for (j, (&o, &n)) in active_set.iter().zip(mi.iter()).enumerate() {
                if o && !n {
                    total_sold += t[j];
                } else if !o && n {
                    new_indices.push(j);
                } else if o && n {
                    x_transferred += s[j];
                }
            }

            let uninformed = 0f64.max(1f64 - x_transferred) / new_indices.len() as f64;
            for j in new_indices {
                s[j] = uninformed;
            }

            active_indices = mi
                .iter()
                .enumerate()
                .filter_map(|(j, &mij)| if mij { Some(j) } else { None })
                .collect::<Vec<usize>>();

            y = Array1::from_iter(active_indices.iter().map(|&j| t[j]));

            y[0] += total_sold * (1f64 - cost);
            y /= y.scalar_sum();

            x = Array1::from_iter(active_indices.iter().map(|&j| s[j]));
            x /= x.scalar_sum(); // if there are no new indices, the 'uninformed' weight never gets allocated.

            active_set = mi.to_owned();

            // method.reset(active_indices.len());

            total_sold
        } else {
            0f64
        };
        // do a step, assume x fits
        let ri = Array1::from_iter(active_indices.iter().map(|&j| r[(i, j)]));
        let step_result =
            StepResult::step(y.view_mut(), x.view_mut(), ri.view(), cost, &mut method)?;
        out[(i, 0)] = step_result.gross_growth;
        out[(i, 1)] = step_result.cash;
        out[(i, 2)] = step_result.transacted + manually_transacted * cost;
        out[(i, 3)] = reset_count;
        for (&j, &xj) in active_indices.iter().zip(x.iter()) {
            out[(i, 4 + j)] = xj;
        }
    }
    Ok(out)
}

pub fn step_all<M>(
    cost: f64,
    x0: ArrayView1<f64>,
    data: ArrayView2<f64>,
    mut method: M,
) -> Result<Array2<f64>, Error>
where
    M: Reset + Step,
{
    if x0.len() != data.dim().1 {
        return Err(Error::new("dimensions of data and x0 must agree"));
    }
    // let mut gd = method.reset(x0.len());
    let mut x = x0.to_owned();
    let mut y = x0.to_owned();
    let mut out = Array2::zeros((data.shape()[0], 3 + x0.len()));
    for (r, mut o) in data.outer_iter().zip(out.outer_iter_mut()) {
        let step_result = StepResult::step(y.view_mut(), x.view_mut(), r, cost, &mut method)?;
        o[0] = step_result.gross_growth;
        o[1] = step_result.cash;
        o[2] = step_result.transacted;
        o.slice_mut(s![3..]).assign(&x);
    }
    Ok(out)
}

pub fn step_constituents_fixed(
    cost: f64,
    x0: ArrayView1<f64>,
    r: ArrayView2<f64>,
    m: ArrayView2<bool>,
) -> Result<Array2<f64>, Error> {
    let (n_obs, n_assets) = m.dim();
    if r.dim().0 != n_obs || r.dim().1 != n_assets || x0.len() != n_assets {
        return Err(Error::new("dimensions of r, m and x0 must agree"));
    }
    let mut out = Array2::ones((n_obs, 3));
    let mut x = Array1::from_elem(0, 0f64);
    let mut active_set = Array1::from_elem(n_assets, false);
    let mut active_indices = Vec::new();

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
            let mut y = Array1::from_iter(new_active_indices.iter().map(|&i| x0[i]));
            y /= y.scalar_sum();

            // volume required to alter the strategy:
            if i > 0 {
                let mut mock_x = Array1::zeros(n_assets);
                x.iter()
                    .zip(&active_indices)
                    .for_each(|(&xi, &i)| mock_x[i] = xi);
                let mut mock_y = Array1::zeros(n_assets);
                y.iter()
                    .zip(&new_active_indices)
                    .for_each(|(&yi, &i)| mock_y[i] = yi);
                let transacted = transaction_cost(mock_x.view(), mock_y.view(), cost)?;
                out[(i - 1, 2)] = transacted;
            }
            x = y;
            active_set = mi.to_owned();
            active_indices = new_active_indices;
        }
        // do a step, assume x fits
        let ri = Array1::from_iter(active_indices.iter().map(|&j| r[(i, j)]));

        let new_pos = &x * &ri;
        let transacted =
            transaction_cost((&new_pos / new_pos.scalar_sum()).view(), x.view(), cost)?;

        out[(i, 0)] = x.dot(&ri) * (1f64 - cost * transacted);
        out[(i, 1)] = x[0];
        out[(i, 2)] = transacted;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use online_gradient_descent::GradientBuilder;
    use rand::distributions::LogNormal;
    use rand::prelude::*;
    #[test]
    fn test_step_constituents() {
        let (n_periods, n_assets) = (70, 10);
        // let x0 = Array1::ones(K) / K as f64;
        let mut x0 = Array1::zeros(n_assets);
        x0[0] = 1f64;
        let ln = LogNormal::new(1f64, 0.5);
        let r = Array2::from_shape_fn((n_periods, n_assets), |(_, j)| {
            if j == 0 {
                1f64
            } else {
                ln.sample(&mut thread_rng())
            }
        });
        let m = Array2::from_shape_fn(
            (n_periods, n_assets),
            |(_, j)| if j == 0 { true } else { random() },
        );
        // let m = Array2::from_elem((T,K), true);
        let method = GradientBuilder::new(1f64, 0f64, 0.002);
        let a = step_constituents(0.002, r.view(), m.view(), method).unwrap();
        println!("{}", a);
        // println!("{}", r);
        // println!("{}", m);
    }
}
