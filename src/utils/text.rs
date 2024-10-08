use crate::base_array::BaseDataset;
use crate::*;
use counter::Counter;
use indexmap::{IndexMap, IndexSet};
use ndarray::{Array1, ArrayView1};
use phf::phf_set;
use phf::Set;
use std::collections::HashSet;

use ndarray_linalg::Norm;

// https://gist.github.com/sebleier/554280
const DEFAULT_STOPWORDS: Set<&str> = phf_set![
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
];

pub struct CountVectorizer {
    preprocessor: Option<fn(&str) -> Array1<String>>,
    stopwords: Option<Vec<String>>,
    max_df: f64,
    min_df: f64,
    max_features: Option<usize>,
    vocabulary: IndexSet<String>,
}

impl CountVectorizer {
    pub fn new() -> Self {
        Self {
            preprocessor: None,
            stopwords: None,
            max_df: 1.0,
            min_df: 0.0,
            max_features: None,
            vocabulary: IndexSet::default(),
        }
    }

    pub fn preprocessor(self, preprocessor: fn(&str) -> Array1<String>) -> Self {
        Self {
            preprocessor: Some(preprocessor),
            ..self
        }
    }

    pub fn stopwords(self, stops: Vec<String>) -> Self {
        Self {
            stopwords: Some(stops),
            ..self
        }
    }

    pub fn set_df_limits(self, max: Option<f64>, min: Option<f64>) -> Self {
        let mut def_max = 1.0;
        let mut def_min = 0.0;
        if let Some(value) = max {
            assert!(value <= 1.0 && value >= 0.0);
            def_max = value;
        }
        if let Some(value) = min {
            assert!(value >= 0.0 && value <= 1.0 && value < def_max);
            def_min = value;
        }

        Self {
            max_df: def_max,
            min_df: def_min,
            ..self
        }
    }

    pub fn max_features(self, max_features: usize) -> Self {
        Self {
            max_features: Some(max_features),
            ..self
        }
    }

    pub fn set_vocabulary(self, vocabulary: Vec<String>) -> Self {
        let vocabulary = IndexSet::from_iter(vocabulary.into_iter());
        Self { vocabulary, ..self }
    }

    pub fn vocabulary(&self) -> &IndexSet<String> {
        &self.vocabulary
    }

    pub fn fit(&mut self, corpus: ArrayView1<&str>) {
        //check if it already has a vocabulary
        if self.vocabulary.len() > 0 {
            eprintln!("Vocabulary already exists, proceeding to reset");
        }
        let len = corpus.len();
        if self.max_df == 1.0 && self.min_df == 0.0 && self.max_features.is_none() {
            let mut res = IndexSet::new();
            corpus.iter().for_each(|s| {
                let result = self.tokenize(s);
                res.extend(result.into_iter());
            });
            self.vocabulary = res;
            return;
        }
        let mut df_counter: Counter<String, f32> = Counter::new();
        corpus.iter().for_each(|s| {
            let res = self.tokenize(s);
            let set: HashSet<String> = HashSet::from_iter(res.into_iter());
            df_counter.extend(set.into_iter());
        });
        //we're done consuming the corpus, i.e. we've fit it, next is to prune it
        df_counter
            .iter_mut()
            .for_each(|(_, value)| *value = *value / len as f32);
        let mut res = df_counter
            .into_iter()
            .filter(|(_, value)| *value >= self.min_df as f32 && *value <= self.max_df as f32)
            .collect::<IndexMap<String, f32>>();
        res.sort_by(|_, v1, _, v2| f32::total_cmp(&v2, &v1));
        let res = match self.max_features {
            Some(max) => res
                .into_iter()
                .take(max)
                .map(|(k, _)| k)
                .collect::<IndexSet<String>>(),
            None => res
                .into_iter()
                .map(|(k, _)| k)
                .collect::<IndexSet<String>>(),
        };
        self.vocabulary = res;
    }

    pub fn transform(&self, corpus: ArrayView1<&str>) -> Array2<usize> {
        let mut res = Array2::from_elem((corpus.len(), self.vocabulary.len()), 0);
        res.rows_mut()
            .into_iter()
            .enumerate()
            .for_each(|(index, mut row)| {
                //first tokenize and transform the corpus
                let res = self.tokenize(corpus[index]);
                //for each element in res, do magic
                res.iter().for_each(|s| {
                    if let Some(col) = self.vocabulary.get_index_of(s) {
                        row[col] += 1;
                    }
                })
            });

        res
    }

    pub fn fit_transform(&mut self, corpus: ArrayView1<&str>) -> Array2<usize> {
        self.fit(corpus);
        self.transform(corpus)
    }

    pub fn tokenize(&self, input: &str) -> Array1<String> {
        let result: Array1<String>;
        match self.preprocessor {
            Some(func) => {
                result = func(input);
            }
            None => match &self.stopwords {
                Some(stops) => {
                    result = tokenizer(input, Some(stops));
                }
                None => {
                    result = tokenizer(input, None);
                }
            },
        };
        result
    }
}

#[derive(Clone, Copy, Default, Hash)]
pub enum TfIdfNormalizer {
    None,
    #[default]
    L1,
    L2,
}

struct TfIdfTransformer {
    smoothing: bool,
    normalizer: TfIdfNormalizer,
    smooth_idf: bool,
    sublinear_tf: bool,
    idf: Array1<f64>,
}

impl TfIdfTransformer {
    pub fn new() -> Self {
        //there are sklearn defaults
        Self {
            smoothing: false,
            normalizer: TfIdfNormalizer::default(),
            smooth_idf: true,
            sublinear_tf: false,
            idf: Array1::default(0),
        }
    }

    pub fn normalizer(self, normalizer: TfIdfNormalizer) -> Self {
        Self { normalizer, ..self }
    }

    pub fn smoothing(self, smoothing: bool) -> Self {
        Self { smoothing, ..self }
    }

    pub fn smooth_idf(self, smooth_idf: bool) -> Self {
        Self { smooth_idf, ..self }
    }

    pub fn sublinear_tf(self, sublinear_tf: bool) -> Self {
        Self {
            sublinear_tf,
            ..self
        }
    }

    pub fn fit(&mut self, mut count_matrix: Array2<usize>) {
        count_matrix.map_mut(|v| *v as f64);
        let mut nsamples = count_matrix.ncols();
        let mut dfs = Array1::from_elem(nsamples, 0f64);
        count_matrix
            .columns()
            .into_iter()
            .enumerate()
            .for_each(|(index, col)| {
                let result = col
                    .iter()
                    .fold(0f64, |accum, val| accum + (*val != 0) as usize as f64);
                dfs[index] = result;
            });
        dfs += self.smooth_idf as usize as f64;
        nsamples += self.smooth_idf as usize;
        let mut idf = nsamples as f64 / dfs;
        idf.map_inplace(|v| *v = v.log10());
        idf += 1f64;
        self.idf = idf;
    }

    pub fn transform(&self, count_matrix: Array2<usize>) -> Array2<f64> {
        assert_eq!(count_matrix.ncols(), self.idf.len());
        let mut count_matrix = count_matrix.map(|&v| v as f64);
        if self.sublinear_tf {
            count_matrix.map_inplace(|v| *v = v.log10() + 1.0);
        }
        count_matrix
            .rows_mut()
            .into_iter()
            .enumerate()
            .for_each(|(index, mut row)| row.map_inplace(|v| *v = self.idf[index] * *v));
        match self.normalizer {
            TfIdfNormalizer::L1 => count_matrix.rows_mut().into_iter().for_each(|mut row| {
                let mut copy = row.to_owned();
                let l1 = copy.norm_l1();
                copy /= l1;
                row.assign(&copy);
            }),
            TfIdfNormalizer::L2 => count_matrix.rows_mut().into_iter().for_each(|mut row| {
                let mut copy = row.to_owned();
                let l2 = copy.norm_l2();
                copy /= l2;
                row.assign(&copy);
            }),
            TfIdfNormalizer::None => {}
        }
        count_matrix
    }
}

struct TfIdfVectorizer {
    preprocessor: Option<fn(&str) -> Array1<String>>,
    stopwords: Option<Vec<String>>,
    max_df: f64,
    min_df: f64,
    max_features: Option<usize>,
    vocabulary: IndexSet<String>,
    smoothing: bool,
    normalizer: TfIdfNormalizer,
    smooth_idf: bool,
    sublinear_tf: bool,
    idf: Array1<f64>,
}

impl TfIdfVectorizer {
    pub fn new() -> Self {
        Self {
            preprocessor: None,
            stopwords: None,
            max_df: 1.0,
            min_df: 0f64,
            max_features: None,
            vocabulary: IndexSet::default(),
            smoothing: false,
            normalizer: TfIdfNormalizer::L1,
            smooth_idf: true,
            sublinear_tf: false,
            idf: Array1::default(0),
        }
    }

    pub fn normalizer(self, normalizer: TfIdfNormalizer) -> Self {
        Self { normalizer, ..self }
    }

    pub fn smoothing(self, smoothing: bool) -> Self {
        Self { smoothing, ..self }
    }

    pub fn smooth_idf(self, smooth_idf: bool) -> Self {
        Self { smooth_idf, ..self }
    }

    pub fn sublinear_tf(self, sublinear_tf: bool) -> Self {
        Self {
            sublinear_tf,
            ..self
        }
    }

    pub fn preprocessor(self, preprocessor: fn(&str) -> Array1<String>) -> Self {
        Self {
            preprocessor: Some(preprocessor),
            ..self
        }
    }

    pub fn stopwords(self, stops: Vec<String>) -> Self {
        Self {
            stopwords: Some(stops),
            ..self
        }
    }

    pub fn set_df_limits(self, max: Option<f64>, min: Option<f64>) -> Self {
        let mut def_max = 1.0;
        let mut def_min = 0.0;
        if let Some(value) = max {
            assert!(value <= 1.0 && value >= 0.0);
            def_max = value;
        }
        if let Some(value) = min {
            assert!(value >= 0.0 && value <= 1.0 && value < def_max);
            def_min = value;
        }

        Self {
            max_df: def_max,
            min_df: def_min,
            ..self
        }
    }

    pub fn max_features(self, max_features: usize) -> Self {
        Self {
            max_features: Some(max_features),
            ..self
        }
    }

    pub fn set_vocabulary(self, vocabulary: Vec<String>) -> Self {
        let vocabulary = IndexSet::from_iter(vocabulary.into_iter());
        Self { vocabulary, ..self }
    }

    fn build_count_vectorizer(&self) -> CountVectorizer {
        CountVectorizer {
            preprocessor: self.preprocessor.clone(),
            stopwords: self.stopwords.clone(),
            max_df: self.max_df,
            min_df: self.min_df,
            max_features: self.max_features,
            vocabulary: self.vocabulary.clone(),
        }
    }

    fn build_tfidf_transformer(&self) -> TfIdfTransformer {
        TfIdfTransformer {
            smoothing: false,
            normalizer: TfIdfNormalizer::default(),
            smooth_idf: true,
            sublinear_tf: false,
            idf: Array1::default(0),
        }
    }
    pub fn vocabulary(&self) -> &IndexSet<String> {
        &self.vocabulary
    }

    pub fn fit(&mut self, corpus: ArrayView1<&str>) {
        let mut vectorizer = self.build_count_vectorizer();
        vectorizer.fit(corpus);
        self.vocabulary = vectorizer.vocabulary().clone();
    }

    pub fn transform(&self, corpus: Array1<&str>) -> Array2<f64> {
        let vectorizer = self.build_count_vectorizer();
        let intermediate = vectorizer.transform(corpus.view());
        let tftransformer = self.build_tfidf_transformer();
        tftransformer.transform(intermediate)
    }

    pub fn fit_transform(&mut self, corpus: ArrayView1<&str>) -> Array2<f64> {
        let mut vectorizer = self.build_count_vectorizer();
        let tftransformer = self.build_tfidf_transformer();
        let intermediate = vectorizer.fit_transform(corpus);
        self.vocabulary = vectorizer.vocabulary().clone();
        tftransformer.transform(intermediate)
    }
}

fn tokenizer(input: &str, stopwords: Option<&Vec<String>>) -> Array1<String> {
    fn not_alphabetic(input: char) -> bool {
        char::is_ascii_punctuation(&input) || char::is_whitespace(input)
    }
    let vector = input
        .split(not_alphabetic)
        .map(|s| s.to_lowercase())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    let collected = match stopwords {
        Some(stops) => {
            let set: HashSet<String> = HashSet::from_iter(stops.iter().map(|s| s.to_string()));
            vector
                .iter()
                .filter(|&x| !set.contains(&x.to_string()))
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        }
        None => vector
            .iter()
            .filter(|&x| !DEFAULT_STOPWORDS.contains(x))
            .map(|s| s.to_string())
            .collect(),
    };
    Array1::from_vec(collected)
}

#[cfg(test)]
mod test {
    use ndarray::array;

    use super::*;
    #[test]
    fn test_tokenizer() {
        let input = "The world shall know Pain. For ages the village of the leaf has oppressed and went after other villages. They shall know my Pain! 'Shinra Tensei!'";
        let res = tokenizer(input, None);
        dbg!(res);
    }

    #[test]
    fn test_count_vectorizer() {
        let corpus = array![
            "I want to get milkshakes at McDonald's sometime in the future",
            "I need tacos from TacoBell",
            "How much is akara sold at Kado Estate?, Since it's cheap near Kado and Jahi",
            "I think we can finish neuron-cml by the beginning of september",
            "Tensorflow is quite the library",
            "I want to enjoy some tacos someday"
        ];
        let mut count = CountVectorizer::new();
        let result = count.fit_transform(corpus.view());
        dbg!(result.view());
        dbg!(&count.vocabulary);
    }
}
