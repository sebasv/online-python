pub use crate::{Error, StepResult};

pub use crate::gradient_descent::GradientDescent;
pub use crate::newton::Newton;
pub use crate::processors::{step_all, step_constituents, step_constituents_fixed};
pub use crate::util::{
    project_simplex, project_simplex_general, transaction_cost, transaction_volume, Grad,
};
