#![feature(specialization)]
extern crate online_python_src;
use online_python_src::prelude as op;
mod version;

extern crate ndarray;
extern crate ndarray_linalg;
extern crate numpy;
// #[macro_use]
extern crate pyo3;

#[cfg(test)]
extern crate rand;

use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::{PyErrArguments, PyErrValue};

struct PythonizableError {
    message: String,
}

impl std::convert::From<op::Error> for PythonizableError {
    fn from(err: op::Error) -> PythonizableError {
        PythonizableError {
            message: err.message,
        }
    }
}

use std::fmt;
impl fmt::Display for PythonizableError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<Rust error>: {}", self.message)
    }
}

macro_rules! impl_to_pyerr {
    ($err: ty, $pyexc: ty) => {
        impl PyErrArguments for $err {
            fn arguments(&self, py: Python) -> PyObject {
                self.to_string().to_object(py)
            }
        }

        impl std::convert::From<$err> for PyErr {
            fn from(err: $err) -> PyErr {
                PyErr::from_value::<$pyexc>(PyErrValue::ToArgs(Box::new(err)))
            }
        }
    };
}
impl_to_pyerr!(PythonizableError, pyo3::exceptions::ValueError);

fn to_pyresult_vec<F, D>(r: Result<F, op::Error>, py: Python) -> PyResult<Py<PyArray<f64, D>>>
where
    D: Dimension,
    F: Into<Array<f64, D>>,
{
    Ok(r.map_err(PythonizableError::from)?
        .into()
        .into_pyarray(py)
        .to_owned())
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
        to_pyresult_vec(op::transaction_volume(w.as_array(), x.as_array(), cost), py)
    }

    /// fn gradient(y, x, r, lambda, cost, utility)
    #[pyfn(m, "gradient")]
    fn gradient(
        py: Python,
        y: &PyArray1<f64>,
        x: &PyArray1<f64>,
        r: &PyArray1<f64>,
        lambda: f64,
        cost: f64,
        utility: &str,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let g = grad(utility)?;
        to_pyresult_vec(
            g.grad(y.as_array(), x.as_array(), r.as_array(), lambda, cost),
            py,
        )
    }

    /// fn project(x)
    /// x: vector to be projected
    /// x is projected in-place.
    #[pyfn(m, "project_simplex")]
    fn project_py(_py: Python, x: &PyArray1<f64>) -> PyResult<()> {
        op::project_simplex(x.as_array_mut())
            .map_err(PythonizableError::from)
            .map_err(PyErr::from)
    }

    /// fn project_simplex_general(x, pos_def_inv, max_iter)
    /// x: vector to be projected
    /// pos_def_inv: the inverted projection matrix (which should be positive definite)
    /// max_iter: the maximum number of iterations before the projection attempt is terminated
    #[pyfn(m, "project_simplex_general")]
    fn project_simplex_general_py(
        py: Python,
        x: &PyArray1<f64>,
        pos_def: &PyArray2<f64>,
        max_iter: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        to_pyresult_vec(
            op::project_simplex_general(x.as_array(), pos_def.as_array(), max_iter),
            py,
        )
    }

    /// fn step_constituents_fixed(cost, x0, r, m) -> out
    /// cost: the transaction cost
    /// x0: fixed allocation
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
        to_pyresult_vec(
            op::step_constituents_fixed(cost, x.as_array(), r.as_array(), m.as_array()),
            py,
        )
    }

    m.add_class::<GradientDescent>()?;
    m.add_class::<Newton>()?;
    m.add_class::<GMV>()?;

    Ok(())
}

fn grad(utility: &str) -> PyResult<op::Grad> {
    match utility {
        "power" => Ok(op::Grad::Power),
        "exp" => Ok(op::Grad::Exp),
        "quad" => Ok(op::Grad::Quad),
        "gmv" => Ok(op::Grad::Gmv),
        "lin" => Ok(op::Grad::Lin),
        _ => Err(PyErr::new::<pyo3::exceptions::ValueError, _>(
            "Did not recognise utility function. Pick one of {'power','exp','quad'}",
        )),
    }
}

#[pyclass]
struct GMV {
    // op: op::GMV,
}

#[pymethods]
impl GMV {
    /// GMV(epsilon, max_iter, n)
    #[new]
    fn __new__(obj: &PyRawObject) -> PyResult<()> {
        obj.init(|_| GMV {
            // op: op::GMV::new(eps, max_iter, n),
        })
    }

    /// fn step_all(eps, positive, cost, x0, data, max_iter) -> results
    /// results: [growth, bank account, transacted]
    #[staticmethod]
    fn step_all(
        py: Python,
        eps: f64,
        positive: bool,
        cost: f64,
        x0: &PyArray1<f64>,
        data: &PyArray2<f64>,
        max_iter: usize,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let n = data.dims()[1];
        to_pyresult_vec(
            op::step_all(
                cost,
                x0.as_array(),
                data.as_array(),
                op::GMV::new(eps, max_iter, n, positive),
            ),
            py,
        )
    }
}

#[pyclass]
struct GradientDescent {
    gd: op::GradientDescent,
    cost: f64,
}

#[pymethods]
impl GradientDescent {
    /// GradientDescent(alpha, gamma, cost, utility)
    #[new]
    fn __new__(
        obj: &PyRawObject,
        alpha: f64,
        lambda: f64,
        cost: f64,
        utility: &str,
    ) -> PyResult<()> {
        let g = grad(utility)?;
        obj.init(|_| GradientDescent {
            gd: op::GradientDescent::new(alpha, lambda, cost, g),
            cost,
        })
    }

    /// fn step(x, r) -> [gross_growth, cash, transacted]
    fn step(
        &mut self,
        py: Python,
        y: &PyArray1<f64>,
        x: &PyArray1<f64>,
        r: &PyArray1<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        to_pyresult_vec(
            op::StepResult::step(
                y.as_array_mut(),
                x.as_array_mut(),
                r.as_array(),
                self.cost,
                &mut self.gd,
            ),
            py,
        )
    }

    /// fn step_constituents(a, lambda, x0, r, m, method, utility) -> out
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
        utility: &str,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let g = grad(utility)?;
        to_pyresult_vec(
            op::step_constituents(
                cost,
                r.as_array(),
                m.as_array(),
                op::GradientDescent::new(a, lambda, cost, g),
            ),
            py,
        )
    }

    /// fn step_all(a,lambda, cost, x0, data, utility) -> results
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
        utility: &str,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let g = grad(utility)?;
        to_pyresult_vec(
            op::step_all(
                cost,
                x0.as_array(),
                data.as_array(),
                op::GradientDescent::new(a, lambda, cost, g),
            ),
            py,
        )
    }
}

#[pyclass]
struct Newton {
    gd: op::Newton,
    cost: f64,
}

#[pymethods]
impl Newton {
    /// Newton(beta, eps, n, max_iter, gamma, cost, utility, n)
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
        utility: &str,
    ) -> PyResult<()> {
        let g = grad(utility)?;
        obj.init(|_| Newton {
            gd: op::Newton::new(beta, max_iter, gamma, cost, g, n),
            cost,
        })
    }

    /// fn step(y, x, r) -> [gross_growth, cash, transacted]
    fn step(
        &mut self,
        py: Python,
        y: &PyArray1<f64>,
        x: &PyArray1<f64>,
        r: &PyArray1<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        to_pyresult_vec(
            op::StepResult::step(
                y.as_array_mut(),
                x.as_array_mut(),
                r.as_array(),
                self.cost,
                &mut self.gd,
            ),
            py,
        )
    }

    fn hess(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(self.gd.approx_hessian.clone().into_pyarray(py).to_owned())
    }

    /// fn step_constituents(beta, max_iter, lambda, cost, r, m, utility) -> out
    #[staticmethod]
    fn step_constituents(
        py: Python,
        beta: f64,
        max_iter: usize,
        lambda: f64,
        cost: f64,
        r: &PyArray2<f64>,
        m: &PyArray2<bool>,
        utility: &str,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let g = grad(utility)?;
        to_pyresult_vec(
            op::step_constituents(
                cost,
                r.as_array(),
                m.as_array(),
                op::Newton::new(beta, max_iter, lambda, cost, g, 1),
            ),
            py,
        )
    }

    /// fn step_all(beta, max_iter, lambda, cost, x0, data, utility) -> results
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
        utility: &str,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let n = x0.shape()[0];
        let g = grad(utility)?;
        to_pyresult_vec(
            op::step_all(
                cost,
                x0.as_array(),
                data.as_array(),
                op::Newton::new(beta, max_iter, lambda, cost, g, n),
            ),
            py,
        )
    }
}
