extern crate online_python_src;
use online_python_src::prelude::*;

extern crate rand;
use rand::distributions::{Distribution, LogNormal};
// use rand::prelude::*;

extern crate ndarray;
use ndarray::prelude::*;

use std::thread;
use std::sync::mpsc;
use std::time::Duration;


fn main() {
    let n = 10;
    let t = 100;
    let cost = 0.002;
    let x0 = Array1::ones(n) / n as f64;
    let beta = 1e-1;
    let max_iter = 100;
    let lambda = 1.0;

    // mean 2, standard deviation 3
    let log_normal = LogNormal::new(0.0002, 0.002);
    let mut rng = rand::thread_rng();
    let data = Array2::from_shape_fn((t, n), |_| log_normal.sample(&mut rng));

    let (tx, rx) = mpsc::channel();
    let clock = thread::spawn(move || {
        let start = std::time::SystemTime::now();
        loop {
            match rx.recv_timeout(Duration::from_secs(1)) {
                Ok(_) => {println!("finished");break},
                Err(_) => println!("{:.2}", start.elapsed().unwrap().as_secs()),
            }
        }
    });
    

    let all = step_all(
        cost,
        x0.view(),
        data.view(),
        NewtonBuilder::new(beta, max_iter, lambda, cost),
    )
    .unwrap();
    tx.send("hang up!");
    clock.join();
    println!("Hello, world!\n{}", all[(0, 0)]);
}
