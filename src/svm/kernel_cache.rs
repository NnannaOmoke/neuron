use crate::svm::RawSVC;
use crate::*;
use lru::LruCache;

use std::num::NonZeroUsize;

const AUTO_CACHE_SIZE_LIMIT: usize = 8_333_334;

#[derive(Hash, PartialEq, PartialOrd, Debug, Eq, Ord, Copy, Clone)]
pub struct IndexPair {
    value: [usize; 2],
}

impl From<[usize; 2]> for IndexPair {
    fn from(mut pair: [usize; 2]) -> Self {
        pair.sort_unstable();
        IndexPair { value: pair }
    }
}

#[derive(Debug, Clone)]
//caches kernel_op_1ds
pub struct KernelCache {
    internal: LruCache<IndexPair, f64>,
}

impl KernelCache {
    pub fn new(cap: usize) -> Self {
        assert!(cap != 0);
        Self {
            internal: LruCache::new(NonZeroUsize::new(cap).unwrap()),
        }
    }

    pub fn new_from_feature_size(feature_array: ArrayView2<f64>) -> Self {
        //just set the cap to the size of all the entries * 2
        //it's 24 bytes per observation, so we can set a hard_limit of ~200Mbs, or 8,333,334 observations
        if feature_array.nrows().pow(2) > AUTO_CACHE_SIZE_LIMIT {
            Self {
                internal: LruCache::new(NonZeroUsize::new(AUTO_CACHE_SIZE_LIMIT).unwrap()),
            }
        } else {
            Self {
                internal: LruCache::new(NonZeroUsize::new(feature_array.nrows().pow(2)).unwrap()),
            }
        }
    }

    pub(crate) fn get(&mut self, indices: IndexPair, svm: &RawSVC) -> f64 {
        *self.internal.get_or_insert(indices, || {
            svm.kernel_op_helper(indices.value[0], indices.value[1])
        })
    }
    pub(crate) fn default(cap: usize) -> Self {
        KernelCache::new(cap)
    }
}

impl Default for KernelCache {
    fn default() -> Self {
        KernelCache::new(1)
    }
}
