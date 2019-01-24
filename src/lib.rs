#![feature(specialization)]
mod online_gradient_descent;
mod online_newton;
mod processors;
mod util;
mod version;

#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate numpy;
extern crate pyo3;

#[cfg(test)]
extern crate rand;

use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

pub trait Build {
    type BuildResult: Step;
    fn build(&self, n: usize) -> Self::BuildResult;
}

pub trait Step {
    fn step(&mut self, x: ArrayViewMut1<f64>, r: ArrayView1<f64>) -> Result<(), Error>;
}

pub struct StepResult {
    gross_growth: f64,
    transacted: f64,
    cash: f64,
}

impl StepResult {
    fn step<S>(
        mut x: ArrayViewMut1<f64>,
        r: ArrayView1<f64>,
        cost: f64,
        stepper: &mut S,
    ) -> Result<StepResult, Error>
    where
        S: Step,
    {
        let mut xr = &x * &r;
        let gross_growth = xr.scalar_sum();
        let cash = x[0];
        xr /= gross_growth;

        stepper.step(x.view_mut(), r)?;

        let transacted = util::transaction_cost(xr.view(), x.view(), cost)?;
        Ok(StepResult {
            gross_growth,
            cash,
            transacted,
        })
    }
}

#[derive(Debug)]
pub enum Error {
    NaNError(&'static str),
    ContiguityError(&'static str),
    SolveError(&'static str),
    InvalidMethodError(&'static str),
    ConvergenceError(&'static str),
}

impl std::convert::From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        match err {
            Error::NaNError(x) => pyo3::exceptions::ValueError::py_err(x),
            Error::ContiguityError(x) => pyo3::exceptions::ValueError::py_err(x),
            Error::SolveError(x) => pyo3::exceptions::ValueError::py_err(x),
            Error::InvalidMethodError(x) => pyo3::exceptions::ValueError::py_err(x),
            Error::ConvergenceError(x) => pyo3::exceptions::ValueError::py_err(x),
        }
    }
}

/// The module docstring
#[pymodinit]
fn online_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", version::PY_VERSION)?;

    /// fn transaction_volume(w, x, cost)
    /// w: current distribution
    /// x: desired distribution
    /// cost: transaction costs
    /// returns a: amount sold
    ///
    /// The total transaction costs are cost*np.abs(a).sum().
    /// The amount of cash sold is cost*np.abs(a).sum() - a.sum().
    #[pyfn(m, "transaction_volume")]
    fn transaction_volume(
        py: Python,
        w: &PyArray1<f64>,
        x: &PyArray1<f64>,
        cost: f64,
    ) -> PyResult<Py<PyArray1<f64>>> {
        Ok(util::transaction_volume(w.as_array(), x.as_array(), cost)?
            .into_pyarray(py)
            .to_owned())
    }

    /// fn project(x)
    /// x: vector to be projected
    /// x is projected in-place.
    #[pyfn(m, "project_simplex")]
    fn project_py(_py: Python, x: &PyArray1<f64>) -> PyResult<()> {
        util::project_simplex(x.as_array_mut())?;
        Ok(())
    }

    /// fn project_simplex_general(x, pos_def, max_iter)
    /// x: vector to be projected
    /// pos_def: the projection matrix (which should be positive definite)
    /// max_iter: the maximum number of iterations before the projection attempt is terminated
    #[pyfn(m, "project_simplex_general")]
    fn project_simplex_general_py(
        py: Python,
        x: &PyArray1<f64>,
        pos_def: &PyArray2<f64>,
        max_iter: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        Ok(
            util::project_simplex_general(x.as_array(), pos_def.as_array(), max_iter)?
                .into_pyarray(py)
                .to_owned(),
        )
    }

    /// fn step_constituents_fixed(a, lambda, x, r, m) -> out
    /// a: alpha-exp-concavity (1)
    /// lambda: risk aversion
    /// x: fixed allocation
    /// data: matrix of rows of return data
    /// out: growth
    #[pyfn(m, "step_constituents_fixed")]
    fn step_constituents_fixed_py(
        py: Python,
        cost: f64,
        x: &PyArray1<f64>,
        r: &PyArray2<f64>,
        m: &PyArray2<bool>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        Ok(
            processors::step_constituents_fixed(cost, x.as_array(), r.as_array(), m.as_array())?
                .into_pyarray(py)
                .to_owned(),
        )
    }

    m.add_class::<GradientDescent>()?;
    m.add_class::<Newton>()?;

    Ok(())
}

#[pyclass]
struct GradientDescent {
    gd: online_gradient_descent::GradientDescent,
    cost: f64,
}

#[pymethods]
impl GradientDescent {
    /// GradientDescent(alpha, gamma, cost)
    #[new]
    fn __new__(obj: &PyRawObject, alpha: f64, gamma: f64, cost: f64) -> PyResult<()> {
        obj.init(|_| GradientDescent {
            gd: online_gradient_descent::GradientDescent::new(alpha, gamma, cost),
            cost,
        })
    }

    /// fn step(x, r) -> [gross_growth, cash, transacted]
    fn step(
        &mut self,
        py: Python,
        x: &PyArray1<f64>,
        r: &PyArray1<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let step_result =
            StepResult::step(x.as_array_mut(), r.as_array(), self.cost, &mut self.gd)?;
        let mut a = Array1::zeros(3);
        a[0] = step_result.gross_growth;
        a[1] = step_result.cash;
        a[2] = step_result.transacted;
        Ok(a.into_pyarray(py).to_owned())
    }

    /// fn step_constituents(a, lambda, x0, r, m, method) -> out
    /// a: alpha-exp-concavity (1)
    /// lambda: risk aversion
    /// data: matrix of rows of return data
    /// method: {'Newton', 'Gradient'} the method to use
    /// out: growth
    #[staticmethod]
    fn step_constituents(
        py: Python,
        a: f64,
        lambda: f64,
        cost: f64,
        r: &PyArray2<f64>,
        m: &PyArray2<bool>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        Ok(processors::step_constituents(
            cost,
            r.as_array(),
            m.as_array(),
            online_gradient_descent::GradientBuilder::new(a, lambda, cost),
        )?
        .into_pyarray(py)
        .to_owned())
    }

    /// fn step_all(a,lambda, x0, data) -> results
    /// a: alpha-exp-concavity (1)
    /// lambda: risk aversion
    /// x0: starting allocation
    /// data: matrix of rows of return data
    /// results: [growth, bank account, transacted]
    #[staticmethod]
    fn step_all(
        py: Python,
        a: f64,
        lambda: f64,
        cost: f64,
        x0: &PyArray1<f64>,
        data: &PyArray2<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        Ok(processors::step_all(
            cost,
            x0.as_array(),
            data.as_array(),
            online_gradient_descent::GradientBuilder::new(a, lambda, cost),
        )?
        .into_pyarray(py)
        .to_owned())
    }
}

#[pyclass]
struct Newton {
    gd: online_newton::Newton,
    cost: f64,
}

#[pymethods]
impl Newton {
    /// Newton(beta, eps, n, max_iter, gamma, cost)
    /// beta:
    /// eps:
    #[new]
    fn __new__(
        obj: &PyRawObject,
        beta: f64,
        n: usize,
        max_iter: usize,
        gamma: f64,
        cost: f64,
    ) -> PyResult<()> {
        obj.init(|_| Newton {
            gd: online_newton::Newton::new(beta, max_iter, gamma, cost, n),
            cost,
        })
    }

    /// fn step(x, r) -> [gross_growth, cash, transacted]
    fn step(
        &mut self,
        py: Python,
        x: &PyArray1<f64>,
        r: &PyArray1<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let step_result =
            StepResult::step(x.as_array_mut(), r.as_array(), self.cost, &mut self.gd)?;
        let mut a = Array1::zeros(3);
        a[0] = step_result.gross_growth;
        a[1] = step_result.cash;
        a[2] = step_result.transacted;
        Ok(a.into_pyarray(py).to_owned())
    }

    fn hess(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(self.gd.approx_hessian.clone().into_pyarray(py).to_owned())
    }

    /// fn step_constituents(beta, max_iter, lambda, cost, r, m) -> out
    #[staticmethod]
    fn step_constituents(
        py: Python,
        beta: f64,
        max_iter: usize,
        lambda: f64,
        cost: f64,
        r: &PyArray2<f64>,
        m: &PyArray2<bool>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        Ok(processors::step_constituents(
            cost,
            r.as_array(),
            m.as_array(),
            online_newton::NewtonBuilder::new(beta, max_iter, lambda, cost),
        )?
        .into_pyarray(py)
        .to_owned())
    }

    /// fn step_all(beta, max_iter, lambda, cost, x0, data) -> results
    /// beta:
    /// max_iter:
    /// lambda: risk aversion
    /// x0: starting allocation
    /// data: matrix of rows of return data
    /// results: [growth, bank account, transacted]
    #[staticmethod]
    fn step_all(
        py: Python,
        beta: f64,
        max_iter: usize,
        lambda: f64,
        cost: f64,
        x0: &PyArray1<f64>,
        data: &PyArray2<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        Ok(processors::step_all(
            cost,
            x0.as_array(),
            data.as_array(),
            online_newton::NewtonBuilder::new(beta, max_iter, lambda, cost),
        )?
        .into_pyarray(py)
        .to_owned())
    }
}
