use super::Error;
use ndarray::prelude::*;
use ndarray_linalg::SolveC;

/// The gradient of the log-risk adjusted growth:
///     
///     `x.dot(&r).ln() - (1f64 - cost * a.mapv(f64::abs).scalar_sum()).ln() - lambda * x.dot(&r).ln().pow(2)`
///
/// Set cost=0 to eliminate transaction costs,
/// set lambda=0 to eliminate risk adjustment.
#[inline]
pub fn grad(
    x: ArrayView1<f64>,
    r: ArrayView1<f64>,
    lambda: f64,
    cost: f64,
) -> Result<Array1<f64>, Error> {
    let mut x_r = &x * &r;

    x_r /= x_r.scalar_sum() + 1f64 - x.scalar_sum(); // such that the amount invested in cash remains the same
    let a = transaction_volume(x_r.view(), x, cost)?;
    let s = a.mapv(f64::signum);

    let xr = x.dot(&r);
    let xrs = x.dot(&(&s + &r));
    let b = 1f64 - cost * x.dot(&s);
    // r minus c times s' times derivative of (a times x' times r)
    let r_csdaxr = (&r + &(&s * &((xr - cost * xrs) / b - &r) * cost)) / b;

    Ok(&r_csdaxr / (xr * (1f64 - cost * s.dot(&a))) - &r * 2f64 * lambda * xr.ln() / xr)
}

/// To rebalance fractions w_i to fractions x_i at cost cost, we must subtract
/// a_i from w_i such that
///     w_i - a_i = x_i * (1 - cost * a.mapv(f64::abs).scalar_sum()),
/// or, defining s = a.mapv(f64::signum), equivalently
///     w_i - a_i = x_i * (1 - cost * a.dot(&s)).
/// Let d_i = w_i - x_i. Then the above can be written as
///     a = d + x * cost * s.dot(&d) / (1 - cost * s.dot(&x)).
/// s depends on a, but the initial guess s = a.mapv(f64::signum) is pretty close.
/// For random x,w and c=0.2 the while loop is executed once on average.
/// For random x,w and c=0.002 the while loop is executed 0.02 times on average.
///
/// If the weights of w or x do not sum to one, the remainder is considered cash,
/// which can be transacted free of charge.
/// The total transaction costs are cost*a.mapv(f64::abs).scalar_sum().
/// The amount of cash sold is cost*a.mapv(f64::abs).scalar_sum() - a.scalar_sum().
///
/// Returns a NaNError if either w or x contain nans, since the algorithm would loop indefinitely on NaNs.
#[inline]
pub fn transaction_volume(
    w: ArrayView1<f64>,
    x: ArrayView1<f64>,
    cost: f64,
) -> Result<Array1<f64>, Error> {
    let d = &w - &x;
    if d.fold(false, |acc, di| acc || di.is_nan()) {
        return Err(Error::NaNError(
            "cannot calculate transaction costs over vector containing nans",
        ));
    }

    let mut s = d.mapv(f64::signum);
    let mut a = &d + &(&x * cost * s.dot(&d) / (1f64 - cost * s.dot(&x)));
    let mut t = a.mapv(f64::signum);

    while !t.all_close(&s, 0.1) {
        s = t;
        a = &d + &(&x * cost * s.dot(&d) / (1f64 - cost * s.dot(&x)));
        t = a.mapv(f64::signum);
    }
    Ok(a)
}

#[inline]
pub fn transaction_cost(w: ArrayView1<f64>, x: ArrayView1<f64>, cost: f64) -> Result<f64, Error> {
    transaction_volume(w, x, cost).map(|a| a.mapv(f64::abs).scalar_sum())
}

/// Projection onto the simplex. Essentially, this constitutes a projection
/// onto the hyperplane with normal line Array::ones(K)/K followed by a
/// projection into the positive orthant. The fastest way to calculate this is
/// by subtracting an equal amount of all coordinates, an then setting the
/// negative coordinates to 0. A single iteration suffices to find what amount
/// needs to be subtracted.
///
/// Returns a NaNError or ContiguityError if the sorting of x fails.
#[inline]
pub fn project_simplex(mut x: ArrayViewMut1<f64>) -> Result<(), Error> {
    if x.fold(false, |acc, xi| acc || xi.is_nan()) {
        return Err(Error::NaNError("cannot project vector containing nans"));
    }
    let mut y = x.to_owned();
    y.as_slice_mut()
        .ok_or(Error::ContiguityError(
            "cannot project vector that is not contiguous",
        ))?
        // .expect("x is not contiguous, slicing failed")
        .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    // .expect("cannot process nans"));
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
    x.mapv_inplace(|xi| 0f64.max(xi - sub));
    // for xi in x.iter_mut() {
    //     *xi = 0f64.max(*xi - sub);
    // }
    Ok(())
}

/// TODO use tricks so you don't actually have to invert matrices here
pub fn project_simplex_general(
    x: ArrayView1<f64>,
    pos_def: ArrayView2<f64>,
    max_iter: usize,
) -> Result<Array1<f64>, Error> {
    let k = x.dim();
    let iota = Array1::ones(k);
    let pos_def_inv_iota = pos_def.solvec(&iota).map_err(|_| {
        println!("{:?}", pos_def);
        Error::SolveError(&"could not solve A^{-1} i")
    })?;
    let iota_pos_def_inv_iota = iota.dot(&pos_def_inv_iota);
    let mut y = &iota / k as f64;
    let x_i = x.dot(&iota);

    let step = |y: ArrayView1<f64>, m: f64| {
        let l = (2f64 - 2f64 * x_i - (m / &y).dot(&pos_def_inv_iota)) / iota_pos_def_inv_iota;
        let mut g = 2f64 * (&y - &x).dot(&pos_def) - m / &y - l * &iota;
        let h = {
            let mut temp = 2f64 * &pos_def;
            for idx in 0..k {
                temp[(idx, idx)] += m * y[idx].powi(-2);
            }
            temp
        };
        h.solvec_inplace(&mut g)
            .map_err(|_| Error::SolveError(&"could not solve H^{-1} g"))?;
        Ok(g)
    };

    let mut m = 1f64; // TODO educated initial guess
    let mut y_ = x.to_owned();

    for c in 0..max_iter {
        if y.all_close(&y_, 1e-8) {
            eprintln!("n iterations: {}", c);
            break;
        }
        let z = &y - &step(y.view(), m)?;
        if z.fold(true, |acc, &el| acc && el > 0f64) {
            if m > 1e-12 {
                m /= 2f64;
            }
            y_ = y;
            y = z;
        } else {
            m *= 2f64;
        }

        if c == max_iter - 1 {
            eprintln!("n iterations: {}", c)
        }
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_linalg::Solve;
    use rand::prelude::*;
    #[test]
    fn test_general_projection() {
        let pos_def = arr2(&[[5f64, 1f64, 2f64], [1f64, 6f64, 1f64], [2f64, 1f64, 5f64]]);
        let x = arr1(&[1f64, 2f64, 3f64]);
        let y = project_simplex_general(x.view(), pos_def.view(), 1000).unwrap();
        println!("{:?}", y);
        assert!(y.all_close(&arr1(&[0f64, 1f64 / 9f64, 8f64 / 9f64]), 1e-7));
    }

    #[test]
    fn test_projection() {
        let size = 10;
        let mut x = Array1::ones(size) / (size as f64);
        project_simplex(x.view_mut()).unwrap();
        assert!(x.sum() <= 1f64);
        assert!(x.sum() + (size as f64) * ::std::f64::EPSILON >= 1f64);
        assert!(x.fold(&1.0, |a, b| if a < b { a } else { b }) >= &0f64);
    }

    #[test]
    fn test_transaction_cost() {
        let shape = 10;
        let mut w = Array1::from_shape_fn(shape, |_| rand::random::<f64>());
        w /= w.scalar_sum();
        let mut x = Array1::from_shape_fn(shape, |_| rand::random::<f64>());
        x /= x.scalar_sum();
        let c = 0.002;
        let a = transaction_volume(w.view(), x.view(), c).unwrap();
        let cost = c * a.mapv(f64::abs).scalar_sum();
        println!("{}", &(&w - &a) - &(&x * (1f64 - cost)));
        assert!((&w - &a).all_close(&(&x * (1f64 - cost)), 1e-16));
    }
}
