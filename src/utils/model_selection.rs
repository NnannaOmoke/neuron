pub enum TrainTestSplitStrategy{
    None,
    TrainTest(f64, f64),
    TrainTestEval(f64, f64, f64),
}

