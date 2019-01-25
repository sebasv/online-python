extern crate online_python;
use online_python::newton::NewtonBuilder;
use online_python::processors::step_all;

extern crate rand;
use rand::distributions::{Distribution, LogNormal};
use rand::prelude::*;

fn main() {
    let N = 500;
    let T = 100;
    let cost = 0.002;
    let x0 = Array1::ones(N) / N as f64;
    let beta = 1e-1;
    let max_iter = 100;
    let lambda = 1.0;

    // mean 2, standard deviation 3
    let log_normal = LogNormal::new(0.0002, 0.002);
    let data = Array2::from_shape_fn((T, N), |_| log_normal.sample(&mut rand::thread_rng()));
    let all = processors::step_all(
        cost,
        x0.view(),
        data,
        NewtonBuilder::new(beta, max_iter, lambda, cost),
    )
    .unwrap();
    println!("Hello, world!\n{}", all[(0, 0)]);
}
