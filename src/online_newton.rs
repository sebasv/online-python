use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};

pub struct Newton {
    #[allow(non_snake_case)]
    Ai: Array2<f64>,
    #[allow(non_snake_case)]
    A: Array2<f64>,
    beta: f64,
    lambda: f64,
}

impl Newton {
    pub fn new(beta: f64, eps: f64, n: usize, lambda: f64) -> Newton {
        Newton {
            A: Array2::eye(n) * eps,
            Ai: Array2::eye(n) / eps,
            beta: beta,
            lambda: lambda,
        }
    }

    pub fn step(&mut self, mut x: ArrayViewMut1<f64>, r: ArrayView1<f64>) {
        let g = self.grad(x.view(), r);
        self.update_A(g.view());
        x -= &(self.Ai.dot(&g) / self.beta);
        self.project(x);
    }

    fn project(&self, mut x: ArrayViewMut1<f64>) {
        // TODO solve how to efficiently minimize_y (x-y)'self.A(x-y)
    }

    fn grad(&self, x: ArrayView1<f64>, r: ArrayView1<f64>) -> Array1<f64> {
        let xr = x.dot(&r);
        let factor = 1f64 / xr - 2f64 * self.lambda * xr.ln() / xr;
        &r * factor
    }

    #[allow(non_snake_case)]
    fn update_A(&mut self, g: ArrayView1<f64>) {
        let v = g
            .into_shape((g.len(), 1))
            .expect("g reshape, should not fail");

        self.A += &(v.dot(&v.t()));

        let u = self.Ai.dot(&v);
        debug_assert!(u.shape() == v.shape());
        self.Ai -= &(u.dot(&u.t()) / (1f64 + u.dot(&g)));
    }
}
