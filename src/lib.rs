#![feature(specialization)]
mod online_gradient_descent;
mod online_newton;
mod util;

extern crate ndarray;
extern crate numpy;
extern crate pyo3;

#[cfg(test)]
extern crate rand;

// use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;

// // immutable example
// fn axpy(a: f64, x: ArrayViewD<f64>, y: ArrayViewD<f64>) -> ArrayD<f64> {
//     a * &x + &y
// }

// // mutable example (no return)
// fn mult(a: f64, mut x: ArrayViewMutD<f64>) {
//     x *= a;
// }

/// The module docstring
#[pymodinit]
fn online_python(_py: Python, m: &PyModule) -> PyResult<()> {
    // // wrapper of `axpy`
    // #[pyfn(m, "axpy")]
    // fn axpy_py(
    //     py: Python,
    //     a: f64,
    //     x: &PyArrayDyn<f64>,
    //     y: &PyArrayDyn<f64>,
    // ) -> Py<PyArrayDyn<f64>> {
    //     let x = x.as_array();
    //     let y = y.as_array();
    //     axpy(a, x, y).into_pyarray(py).to_owned()
    // }

    // // wrapper of `mult`
    // #[pyfn(m, "mult")]
    // fn mult_py(_py: Python, a: f64, x: &PyArrayDyn<f64>) -> PyResult<()> {
    //     let x = x.as_array_mut();
    //     mult(a, x);
    //     Ok(())
    // }

    /// w: current distribution
    /// x: desired distribution
    /// cost: transaction costs
    /// returns a: amount sold
    ///
    /// The total transaction costs are cost*np.abs(a).sum().
    /// The amount of cash sold is cost*np.abs(a).sum() - a.sum().
    #[pyfn(m, "transaction_cost")]
    fn transaction_cost_py(
        py: Python,
        w: &PyArray1<f64>,
        x: &PyArray1<f64>,
        cost: f64,
    ) -> Py<PyArray1<f64>> {
        util::transaction_cost(w.as_array(), x.as_array(), cost)
            .into_pyarray(py)
            .to_owned()
    }

    /// fn project(x)
    /// x: vector to be projected
    /// x is projected in-place.
    #[pyfn(m, "project_simplex")]
    fn project_py(_py: Python, x: &PyArray1<f64>) -> PyResult<()> {
        util::project_simplex(x.as_array_mut());
        Ok(())
    }

    /// fn step_all(a,lambda, x0, data) -> results
    /// a: alpha-exp-concavity (1)
    /// lambda: risk aversion
    /// x0: starting allocation
    /// data: matrix of rows of return data
    /// results: [growth, bank account, transacted]
    #[pyfn(m, "step_all")]
    fn step_all_py(
        py: Python,
        a: f64,
        lambda: f64,
        x0: &PyArray1<f64>,
        data: &PyArray2<f64>,
    ) -> Py<PyArray2<f64>> {
        online_gradient_descent::step_all(a, lambda, x0.as_array(), data.as_array())
            .into_pyarray(py)
            .to_owned()
    }

    /// fn step_constituents(a, lambda, x0, r, m) -> out
    /// a: alpha-exp-concavity (1)
    /// lambda: risk aversion
    /// x0: starting allocation
    /// data: matrix of rows of return data
    /// out: growth
    #[pyfn(m, "step_constituents")]
    fn step_constituents(
        py: Python,
        a: f64,
        lambda: f64,
        x0: &PyArray1<f64>,
        r: &PyArray2<f64>,
        m: &PyArray2<bool>,
    ) -> Py<PyArray2<f64>> {
        online_gradient_descent::step_constituents(
            a,
            lambda,
            x0.as_array(),
            r.as_array(),
            m.as_array(),
        )
        .into_pyarray(py)
        .to_owned()
    }

    m.add_class::<GradientDescent>()?;
    m.add_class::<Newton>()?;

    Ok(())
}

#[pyclass]
struct GradientDescent {
    gd: online_gradient_descent::GradientDescent,
}

#[pymethods]
impl GradientDescent {
    #[new]
    fn __new__(obj: &PyRawObject, alpha: f64, gamma: f64) -> PyResult<()> {
        obj.init(|_| GradientDescent {
            gd: online_gradient_descent::GradientDescent::new(alpha, gamma),
        })
    }

    fn step(&mut self, x: &PyArray1<f64>, r: &PyArray1<f64>) -> PyResult<()> {
        self.gd.step(x.as_array_mut(), r.as_array());
        Ok(())
    }

    // fn step_all(
    //     &mut self,
    //     py: Python,
    //     x: &PyArray1<f64>,
    //     data: &PyArray2<f64>,
    // ) -> Py<PyArray2<f64>> {
    //     self.gd
    //         .step_all(x.as_array_mut(), data.as_array())
    //         .into_pyarray(py)
    //         .to_owned()
    // }
}

#[pyclass]
struct Newton {
    gd: online_newton::Newton,
}

#[pymethods]
impl Newton {
    #[new]
    fn __new__(obj: &PyRawObject, beta: f64, eps: f64, n: usize, gamma: f64) -> PyResult<()> {
        obj.init(|_| Newton {
            gd: online_newton::Newton::new(beta, eps, n, gamma),
        })
    }

    fn step(&mut self, x: &PyArray1<f64>, r: &PyArray1<f64>) -> PyResult<()> {
        self.gd.step(x.as_array_mut(), r.as_array());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
