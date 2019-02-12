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
    let mut out = Array2::from_elem((n_obs, 3), f64::NAN);
    let mut x = Array1::zeros(n_assets);
    let mut y = Array1::zeros(n_assets);
    let mut active_set = Array1::from_elem(n_assets, false);
    let mut active_indices = Vec::new();

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
            let mut x_new = Array1::ones(warm_start_k) / warm_start_k as f64;
            let mut y_new = Array1::zeros(warm_start_k);
            y_new[0] = 1f64;
            // ... do warm start ...
            let max_start =
                new_active_indices
                    .iter()
                    .fold(0, |acc, &i| if acc < starts[i] { starts[i] } else { acc });

            let warm_start_r = {
                let mut temp = Array2::zeros((i - max_start, warm_start_k));
                for (j_, &j) in new_active_indices.iter().enumerate() {
                    temp.slice_mut(s![.., j_])
                        .assign(&r.slice(s![max_start..i, j]))
                }
                temp
            };

            // reset method
            method.reset(warm_start_k);
            for ri in warm_start_r.outer_iter() {
                let xr = &x_new * &ri;
                method.step(y_new.view(), x_new.view_mut(), ri)?;
                y_new.assign(&(&xr / xr.scalar_sum()));
            }
            // volume required to alter the strategy:
            if i > 0 {
                let mut mock_x = Array1::zeros(n_assets);
                x.iter()
                    .zip(&active_indices)
                    .for_each(|(&xi, &i)| mock_x[i] = xi);
                let mut mock_y = Array1::zeros(n_assets);
                x_new
                    .iter()
                    .zip(&new_active_indices)
                    .for_each(|(&yi, &i)| mock_y[i] = yi);
                let transacted = transaction_cost(mock_x.view(), mock_y.view(), cost)?;
                out[(i, 2)] = transacted;
            }
            x = x_new;
            y = y_new;
            active_set = mi.to_owned();
            active_indices = new_active_indices;
        }
        // do a step, assume x fits
        let ri = Array1::from_iter(active_indices.iter().map(|&j| r[(i, j)]));
        let step_result =
            StepResult::step(y.view_mut(), x.view_mut(), ri.view(), cost, &mut method)?;
        out[(i, 0)] = step_result.gross_growth;
        out[(i, 1)] = step_result.cash;
        if out[(i, 2)].is_nan() {
            // transaction cost haven't been set yet
            out[(i, 2)] = step_result.transacted;
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
    // let mut gd = method.reset(x0.len());
    let mut x = x0.to_owned();
    let mut y = Array1::zeros(x0.len());
    y[0] = 1f64;
    let mut out = Array2::zeros((data.shape()[0], 3));
    for (r, mut o) in data.outer_iter().zip(out.outer_iter_mut()) {
        let step_result = StepResult::step(y.view_mut(), x.view_mut(), r, cost, &mut method)?;
        o[0] = step_result.gross_growth;
        o[1] = step_result.cash;
        o[2] = step_result.transacted;
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
