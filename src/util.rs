use ndarray::prelude::*;

/// The gradient of the log-risk adjusted growth:
///     
///     `x.dot(&r).ln() - (1f64 - cost * a.mapv(f64::abs).scalar_sum()).ln() - lambda * x.dot(&r).ln().pow(2)`
///
/// Set cost=0 to eliminate transaction costs,
/// set lambda=0 to eliminate risk adjustment.
pub fn grad(x: ArrayView1<f64>, r: ArrayView1<f64>, lambda: f64, cost: f64) -> Array1<f64> {
    let mut x_r = &x * &r;
    x_r /= x_r.scalar_sum() + 1f64 - x.scalar_sum(); // such that the amount invested in cash remains the same
    let a = transaction_cost(x_r.view(), x, cost);
    let s = a.mapv(f64::signum);

    let xr = x.dot(&r);
    let factor = 1f64 / xr - 2f64 * lambda * xr.ln().min(0f64) / xr;
    &r * factor + cost * &s / (1f64 - cost * s.dot(&a))
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
/// # Panics
/// 
/// If any vector contains nans.
pub fn transaction_cost(w: ArrayView1<f64>, x: ArrayView1<f64>, cost: f64) -> Array1<f64> {
    let d = &w - &x;
    assert!(d.fold(true, |acc, di| acc && !di.is_nan()));
    let mut s = d.mapv(f64::signum);
    let mut a = &d + &(&x * cost * s.dot(&d) / (1f64 - cost * s.dot(&x)));
    let mut t = a.mapv(f64::signum);
    while !t.all_close(&s, 0.1) {
        s = t;
        a = &d + &(&x * cost * s.dot(&d) / (1f64 - cost * s.dot(&x)));
        t = a.mapv(f64::signum);
    }
    a
}

/// Projection onto the simplex. Essentially, this constitutes a projection
/// onto the hyperplane with normal line Array::ones(K)/K followed by a
/// projection into the positive orthant. The fastest way to calculate this is
/// by subtracting an equal amount of all coordinates, an then setting the
/// negative coordinates to 0. A single iteration suffices to find what amount
/// needs to be subtracted.
pub fn project_simplex(mut x: ArrayViewMut1<f64>) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection() {
        let N = 10;
        let mut x = Array1::ones(N) / (N as f64);
        // let mut x_ = Array1::ones(N) / (N as f64);
        project_simplex(x.view_mut());
        assert!(x.sum() <= 1f64);
        assert!(x.sum() + (N as f64) * ::std::f64::EPSILON >= 1f64);
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
        let a = transaction_cost(w.view(), x.view(), c);
        let cost = c * a.mapv(f64::abs).scalar_sum();
        println!("{}", &(&w - &a) - &(&x * (1f64 - cost)));
        assert!((&w - &a).all_close(&(&x * (1f64 - cost)), 1e-16));
    }
}
