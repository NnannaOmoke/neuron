pub mod math;
pub mod metrics;
pub mod model_selection;
pub mod scaler;

use crate::base_array::BaseDataset;
use std::collections::HashSet;

pub fn get_dummies(
    dataset: BaseDataset,
    exclude: &[&str],
    name: Option<fn(&[&str]) -> Vec<String>>,
) -> BaseDataset {
    let excluded = exclude.iter().collect::<HashSet<_>>()
    let included = dataset.column_names.iter().collect::<HashSet<_>>();
    let targets = included.difference(&included).collect::<Vec<_>>();
    todo!();
    
}
