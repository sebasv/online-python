pub use crate::{Error, StepResult};

pub use crate::gradient_descent::{GradientBuilder, GradientDescent};
pub use crate::newton::{Newton, NewtonBuilder};
pub use crate::processors::{step_all, step_constituents, step_constituents_fixed};
pub use crate::util::{
    grad, project_simplex, project_simplex_general, transaction_cost, transaction_volume,
};
