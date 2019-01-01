#![feature(specialization)]
mod online_gradient_descent;
mod online_newton;

extern crate ndarray;
extern crate numpy;
extern crate pyo3;

// use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::PyArray1;
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
