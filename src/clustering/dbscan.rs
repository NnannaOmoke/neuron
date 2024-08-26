use ndarray::{prelude::*, AssignElem};
use petal_neighbors::{distance::Euclidean, BallTree};
use std::collections::VecDeque;

pub struct DBSCAN {
    epsilon: f64,
    min_points: usize,
    core_points: Array1<usize>,
    labels: Array1<isize>,
}

impl DBSCAN {
    pub fn new(eps: f64, min_points: usize) -> Self {
        Self {
            epsilon: eps,
            min_points,
            core_points: Array1::default(0),
            labels: Array1::default(0),
        }
    }
    fn fit(&mut self, external: ArrayView2<f64>) {
        //first build an internal ball-tree to represent the data
        let tree = BallTree::new(external.to_owned(), Euclidean {}).unwrap();
        let mut radii = Array1::from_elem(external.nrows(), vec![]);
        radii.iter_mut().enumerate().for_each(|(index, point)| {
            let nearest = tree.query_radius(&external.row(index), self.epsilon);
            *point = nearest;
        });
        let npoints = Array1::from_shape_fn(external.nrows(), |index| radii[index].len());
        let mut labels = Array1::from_elem(external.nrows(), -1);
        let core_samples = npoints
            .iter()
            .map(|&count| {
                if count >= self.min_points {
                    true
                } else {
                    false
                }
            })
            .collect::<Array1<bool>>();
        fn inner_dbscan_loop(
            is_core: ArrayView1<bool>,
            radii: ArrayView1<Vec<usize>>,
            labels: &mut ArrayViewMut1<isize>,
        ) {
            let mut curr_label = 0isize;
            let mut v;
            let mut stack = VecDeque::new();
            let mut neighb;

            for index in 0..labels.len() {
                if labels[index] != -1 || !is_core[index] {
                    continue;
                }
                loop {
                    if labels[index] == -1 {
                        labels[index] = curr_label;
                        if is_core[index] {
                            neighb = &radii[index];
                            for (jindex, _) in neighb.iter().enumerate() {
                                v = jindex;
                                if labels[v] == -1 {
                                    stack.push_back(v);
                                }
                            }
                        }
                    }
                    if stack.len() == 0 {
                        break;
                    }
                    stack.pop_back();
                }
                curr_label += 1;
            }
        }

        inner_dbscan_loop(core_samples.view(), radii.view(), &mut labels.view_mut());
        self.core_points = core_samples
            .into_iter()
            .enumerate()
            .filter_map(|(index, value)| if value { Some(index) } else { None })
            .collect();
        self.labels = labels;
    }

    fn fit_predict(&mut self, data: ArrayView2<f64>) -> Array1<isize> {
        self.fit(data);
        self.labels.clone()
    }
}
