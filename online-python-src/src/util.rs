use super::Error;
use ndarray::prelude::*;
use ndarray_linalg::SolveC;
use std::f64;

pub enum Grad {
    /// gradient of
    ///     growth^(1-lambda) - 1
    ///     ---------------------
    ///          1 - lambda
    /// exp-convex only for alpha <= lambda = 1
    Power,

    /// gradient of
    ///     1 - e^(-lambda * growth)
    ///     ------------------------
    ///             lambda
    /// exp-convex for alpha <= lambda
    Exp,

    /// gradient of
    ///     growth - lambda * (growth)^2
    /// never exp-convex
    Quad,

    /// gradient of
    ///     - growth^2
    /// (online approach to global minimum variance)
    Gmv,

    /// gradient of
    ///     growth
    /// the risk-neutral grower.
    Lin,
}

impl Grad {
    pub fn grad(
        &self,
        y: ArrayView1<f64>,
        x: ArrayView1<f64>,
        r: ArrayView1<f64>,
        lambda: f64,
        cost: f64,
    ) -> Result<Array1<f64>, Error> {
        let (growth, d_growth) = prepare_grad(y, x, r, lambda, cost)?;
        match self {
            Grad::Quad => Ok(&d_growth * (1f64 - lambda * 2f64 * growth)),
            Grad::Gmv => Ok(-d_growth * growth),
            Grad::Lin => Ok(d_growth),
            // what are these even?
            Grad::Exp => Ok(&d_growth * (-lambda * growth).exp()),
            Grad::Power => Ok(d_growth * growth.powf(-lambda)),
        }
    }
}

pub fn prepare_grad(
    y: ArrayView1<f64>,
    x: ArrayView1<f64>,
    r: ArrayView1<f64>,
    lambda: f64,
    cost: f64,
) -> Result<(f64, Array1<f64>), Error> {
    if r.len() != x.len() || x.len() != y.len() {
        return Err(Error::new("lengths of r, y and x must match"));
    }
    if lambda < 0f64 {
        return Err(Error::new("lambda cannot be negative"));
    }
    if cost < 0f64 || cost >= 1f64 {
        return Err(Error::new("cost must be from [0,1)"));
    }

    let xr = x.dot(&r);
    let a = transaction_volume(y, x, cost)?;
    let s = a.mapv(f64::signum);
    let csa = cost * s.dot(&a);
    let growth = (1f64 - csa) * xr;
    let d_growth = (1f64 - csa) * (&r + &(&s * cost * xr / (1f64 - cost * s.dot(&x))));
    // Ok((growth, d_growth))
    Ok((growth.ln(), &d_growth / growth))
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
/// Alternative representations of s:
///     s = sign(w(1-cs'x) - x(1-cs'w)) = sign(w - x + c(xw' - wx')s)
///
/// Returns a ValueError if either w or x contain nans, since the algorithm would loop indefinitely on NaNs.
#[inline]
pub fn transaction_volume(
    w: ArrayView1<f64>,
    x: ArrayView1<f64>,
    cost: f64,
) -> Result<Array1<f64>, Error> {
    let d = &w - &x;
    if d.fold(false, |acc, di| acc || di.is_nan()) {
        return Err(Error::new(
            "cannot calculate transaction costs over vector containing nans",
        ));
    }
    if w.len() != x.len() {
        return Err(Error::new("lengths of y and x must match"));
    }
    if cost < 0f64 || cost >= 1f64 {
        return Err(Error::new("cost must be from [0,1)"));
    }

    let mut s = d.mapv(f64::signum);
    let mut a = &w - &(&x * (1f64 - cost * s.dot(&w)) / (1f64 - cost * s.dot(&x)));
    // let mut a = &d + &(&x * cost * s.dot(&d) / (1f64 - cost * s.dot(&x)));
    let mut t = a.mapv(f64::signum);

    while !t.all_close(&s, 0.1) {
        s = t;
        a = &w - &(&x * (1f64 - cost * s.dot(&w)) / (1f64 - cost * s.dot(&x)));
        // a = &d + &(&x * cost * s.dot(&d) / (1f64 - cost * s.dot(&x)));
        t = a.mapv(f64::signum);
    }
    Ok(a)
}

#[inline]
pub fn transaction_cost(w: ArrayView1<f64>, x: ArrayView1<f64>, cost: f64) -> Result<f64, Error> {
    transaction_volume(w, x, cost).map(|a| a.mapv(f64::abs).scalar_sum() * cost)
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
        return Err(Error::new("cannot project vector containing nans"));
    }
    let mut idx = (0..x.len()).collect::<Vec<usize>>();
    idx.sort_unstable_by(|a, b| x[*a].partial_cmp(&x[*b]).unwrap());

    let mut s = x.sum() - 1f64;
    let mut prev = 0f64;
    let mut nrem = x.len();
    let mut iter = idx.iter();
    let mut kahan = 0f64;
    while let Some(&i) = iter.next() {
        let diff = x[i] - prev;
        if s + kahan > diff * nrem as f64 {
            {
                // kahan summation
                let y = -diff * nrem as f64;
                let t = s + y;
                kahan += if s.abs() > y.abs() {
                    (s - t) + y
                } else {
                    (y - t) + s
                };
                s = t;
            }
            prev = x[i];
            nrem -= 1;
            x[i] = 0f64;
        } else {
            let sub = iter.clone().fold(x[i] - 1f64, |acc, &ii| acc + x[ii]) / nrem as f64;
            x[i] = 0f64.max(x[i] - sub);
            while let Some(&i) = iter.next() {
                x[i] = 0f64.max(x[i] - sub);
            }
        }
    }
    Ok(())
}

/// Calculate the generalized projection of x on the simplex:
/// returns argmin_y (y-x)'A(y-x) s.t. y>=0, sum(y)=1.
///
/// The algorithm is an active set method. The first order conditions of the lagrangian
///     (y-x)'A(y-x) - lambda(sum(x)-1) - mu'x.
/// are solved in each iteration, where lambda is always tight and mu is determined based
/// on the active set from the previous iteration.
///
/// These foc can be written in the form x = a + B'mu. For the inactive x_i, mu_i is 0.
/// For the active x_i we must have 0 = a_i + B_{i,:}'mu.
/// This yields n equations from which mu can be solved, after which x can be solved.
pub fn project_simplex_general(
    y: ArrayView1<f64>,
    pos_def_inv: ArrayView2<f64>,
    max_iter: usize,
) -> Result<Array1<f64>, Error> {
    let n = y.dim();
    let (n1, n2) = pos_def_inv.dim();
    if n1 != n || n2 != n {
        return Err(Error::new(
            "pos_def must be square and must match the dimensions of x",
        ));
    }
    let ipd = pos_def_inv.sum_axis(Axis(1)); //dot(&Array1::ones(n));
    let ipd2 = ipd.to_owned().insert_axis(Axis(1)); //pos_def_inv.dot(&Array2::ones((n, 1)));
    let ipdi = ipd.scalar_sum();

    let a = &y + &((1f64 - y.scalar_sum()) / ipdi * &ipd);
    if a.fold(1f64, |acc, &ai| acc.min(ai)) >= 0f64 {
        // eprintln!("unconstrained problem satisfactory");
        return Ok(a);
    }
    let a_sum = a.scalar_sum();
    let b = &pos_def_inv - &(ipd2.dot(&ipd2.t()) / ipdi);

    // Initialize without the >=0 constraint
    let mut m = a.mapv(|xi| xi <= 0f64);
    let mut x = a.to_owned();
    for _count in 0..max_iter {
        let actives = m
            .iter()
            .enumerate()
            .filter_map(|(i, &mi)| if mi { Some(i) } else { None })
            .collect::<Vec<usize>>();
        let inactives = m
            .iter()
            .enumerate()
            .filter_map(|(i, &mi)| if !mi { Some(i) } else { None })
            .collect::<Vec<usize>>();

        let a_k = a.select(Axis(0), &actives);
        let b_nk = b.select(Axis(1), &actives);
        let b_kk = b_nk.select(Axis(0), &actives);
        let b_n_kk = b_nk.select(Axis(0), &inactives);

        let mu = b_kk.solvec(&a_k).unwrap_or_else(|e| {
            panic!(
                "could not solve the active-set lagrange multipliers: {}\n{}\n{}",
                e, b_kk, a_k,
            )
        });

        let proj = b_n_kk.dot(&mu);
        let correction = (1f64 - a_sum + a_k.scalar_sum() + proj.scalar_sum()) / proj.len() as f64;

        // set the xi with active constraints 0
        actives.iter().for_each(|&i| x[i] = 0f64);
        // set the xi with inactice constraints to their calculated value
        // Apply a correction to counter round-off errors in the matrix inversion etc
        inactives
            .iter()
            .zip(proj.iter())
            .for_each(|(&i, &pi)| x[i] = a[i] - pi + correction);

        // Check the KKT conditions: primal && dual feasibility
        if x.fold(1f64, |acc, &xi| acc.min(xi)) >= 0f64
            && mu.fold(-1f64, |acc, &mui| acc.max(mui)) <= 0f64
        {
            return Ok(x);
        }

        // update the active set: Take all the xi that are at their constraint or violate the constraint
        m = x.mapv(|xi| xi <= 0f64);
        // remove those elements from the active set with invalid lagrange multipliers
        actives
            .iter()
            .zip(mu.iter())
            .for_each(|(&i, &mui)| m[i] &= mui <= 0f64);
    }
    eprintln!("reached max count");
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn test_choleski() {
        let a: Array2<f64> = Array2::eye(3);
        let b = a.factorizec(UPLO::Upper).unwrap();
        // println!("{:?}", b);
    }

    #[test]
    fn test_grad() {
        let x = arr1(&[0.1, 0.2, 0.3, 0.4]);
        let r = arr1(&[0.9, 0.95, 1.0, 1.05]);
        let lambda = 1.0;
        let cost = 0.002;
        let f = |x: ArrayView1<f64>, r: ArrayView1<f64>, lambda, cost| {
            let xr = x.dot(&r);
            let x_r = &x * &r;
            let sa = transaction_cost((x_r / xr).view(), x, cost).unwrap();
            let lnxr = xr.ln();
            lnxr + (1. - cost * sa).ln() - lambda * lnxr.powi(2)
        };

        let approx_grad = {
            let mut e = x.to_owned();
            let f0 = f(e.view(), r.view(), lambda, cost);
            let eps = 1e-8;
            let mut out = Array1::zeros(e.len());
            for (i, mut oi) in out.iter_mut().enumerate() {
                e[i] += eps;
                *oi = (f(e.view(), r.view(), lambda, cost) - f0) / eps;
                e[i] -= eps;
            }
            out
        };
        let anal_grad = grad(x.view(), r.view(), lambda, cost).unwrap();
        println!(
            "{}\n{}\n{}",
            approx_grad,
            anal_grad,
            &approx_grad - &anal_grad
        );
        assert!((&anal_grad - &approx_grad).mapv(f64::abs).scalar_sum() < 1e-6);
    }
}
