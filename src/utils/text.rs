use crate::base_array::BaseDataset;
use crate::*;
use counter::Counter;
use indexmap::IndexSet;
use ndarray::{Array1, ArrayView1};
use phf::phf_set;
use phf::Set;
use std::collections::HashSet;

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
    max_features: Option<u32>,
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

    pub fn max_features(self, max_features: u32) -> Self {
        Self {
            max_features: Some(max_features),
            ..self
        }
    }

    pub fn set_vocabulary(self, vocabulary: Vec<String>) -> Self {
        let vocabulary = IndexSet::from_iter(vocabulary.into_iter());
        Self { vocabulary, ..self }
    }

    pub fn fit(&mut self, corpus: ArrayView1<&str>) {
        //check if it already has a vocabulary
        if self.vocabulary.len() > 0 {
            eprintln!("Vocabulary already exists, proceeding to reset");
        }
        let len = corpus.len();
        if self.max_df == 1.0 && self.min_df == 0.0 {
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
            .map(|(key, _)| key)
            .collect::<Vec<String>>();
        res.sort_unstable();
        let res = res.into_iter().collect::<IndexSet<String>>();
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

struct TfIdfVectorizer {}

struct TfIdfTranformer {}

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
            "Tensorflow is quite the library"
        ];
        let mut count = CountVectorizer::new().set_df_limits(Some(0.95), None);
        let result = count.fit_transform(corpus.view());
        dbg!(result.view());
        dbg!(&count.vocabulary);
    }
}
