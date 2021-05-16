//! This crate implements a simple caching structure
use crate::Ttl::{Bounded, Unbounded};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Add;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

#[doc(hidden)]
pub trait Now {
    fn now(&self) -> Instant;
}

#[doc(hidden)]
#[derive(Clone)]
pub struct InstantNow;

impl Now for InstantNow {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Ttl {
    Unbounded,
    Bounded(Instant),
}

#[derive(Clone, PartialEq, Eq)]
struct CachedValue<V> {
    ttl: Ttl,
    value: V,
}

impl<V> CachedValue<V> {
    fn new(ttl: Ttl, value: V) -> Self {
        CachedValue { ttl, value }
    }
}

pub(crate) struct InnerSimpleCache<K, V, N: Now> {
    items: HashMap<K, CachedValue<V>>,
    now: N,
}

/// A simple multithreaded cache.
///
/// The cache is implemented as a simple [RwLock] with [HashMap].
///
/// Implementation details and considerations:
/// * For simplicity sake, expired entries are removed upon the next retrieval. \
///   Only if the value is expired, a write lock is acquired, so when the expiry feature is not used, parallel reads are possible. \
///   It would make sense to add a function that scans for expired keys and removes them to cleanup without remembering the keys, or
///   alternatively expose the internal [HashMap] iterator to walk over all entries.
/// * Memory usage is equivalent to HashMap memory usage, plus a wrapper around the value to store [Ttl]
/// * There is no limit on the cache size, so no LRU or other expiry mechanism is implemented.
///
/// The required performance targets are:
/// * Retrieving a key within 1ms for 95th percentile
/// * Retrieving a key within 5ms for 99th percentile
/// * Handle up to 10_000_000 key/value pairs
///
/// Some optimizations options would be:
/// * Add a proper benchmark with Criterion
/// * Use Dashmap or Cht as it aims to be a simple replacement for our design. \
///   In general a concurrent hashmap (as available in JDK / Java) would be a bit faster if more threads use the cache.
/// * Don't delete an expired item on lookup, but do it periodically to lower contention.
/// * Use Quanta to optimize the calculation of the duration, as it supports updating the time on a background thread, which decreases overhead of getting the time.
///
#[derive(Clone)]
pub struct SimpleCache<K, V, N = InstantNow>
where
    N: Now,
{
    inner: Arc<RwLock<InnerSimpleCache<K, V, N>>>,
}

impl<K, V> Default for SimpleCache<K, V>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        SimpleCache::new()
    }
}

impl<K, V> SimpleCache<K, V>
where
    K: Eq + Hash,
{
    pub fn new() -> Self {
        SimpleCache {
            inner: Arc::new(RwLock::new(InnerSimpleCache {
                items: HashMap::new(),
                now: InstantNow,
            })),
        }
    }

    /// Put a key with value in the cache, returns old value if available and not expired.
    pub fn cache(&self, key: K, value: V) -> Option<V> {
        let mut inner = self.inner.write().unwrap();
        let previous = inner.items.remove(&key).map(|v| v.value);
        inner.items.insert(key, CachedValue::new(Unbounded, value));
        previous
    }

    /// Put a key with value in the cache with a time-to-live, returns old value if available and
    /// not expired.
    pub fn cache_ttl(&self, key: K, value: V, ttl: Duration) -> Option<V> {
        let mut inner = self.inner.write().unwrap();
        let previous = inner.items.remove(&key).map(|v| v.value);
        let now: Instant = inner.now.now();
        let expiry = now.add(ttl);
        inner
            .items
            .insert(key, CachedValue::new(Bounded(expiry), value));
        previous
    }

    /// Remove a key from the cache, returns value if available.
    pub fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash,
    {
        let mut inner = self.inner.write().unwrap();
        inner.items.remove(key).map(|v| v.value)
    }

    /// Retrieve a key from the cache, clones the value.
    pub fn retrieve<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash,
        V: Clone,
    {
        let mut found_expired = false;
        let result;
        {
            let inner = self.inner.read().unwrap();
            result = inner.items.get(key).and_then(|v| match v.ttl {
                Unbounded => Some(v.value.clone()),
                Bounded(instant) => {
                    if inner.now.now() <= instant {
                        Some(v.value.clone())
                    } else {
                        found_expired = true;
                        None
                    }
                }
            });
        }
        if found_expired {
            self.remove(key);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::SimpleCache;
    use std::ops::Add;
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn simple_roundtrip() {
        let cache = SimpleCache::new();
        cache.cache("a", "a");
        assert_eq!("a", cache.retrieve(&"a").unwrap());
    }

    #[test]
    fn overwrite_gets_previous() {
        let cache = SimpleCache::new();
        cache.cache("a", "a");
        assert_eq!("a", cache.cache("a", "b").unwrap());
    }

    #[test]
    fn remove_test() {
        let cache = SimpleCache::new();
        cache.cache("a", "a");
        cache.remove(&"a");
        assert_eq!(None, cache.retrieve(&"a"));
        assert_eq!(None, cache.cache("a", "a"));
    }

    #[test]
    fn simple_ttl_test() {
        let cache = SimpleCache::new();
        cache.cache_ttl("a", "a", Duration::from_millis(10));
        let first = cache.retrieve(&"a");
        thread::sleep(Duration::from_millis(20));
        let second = cache.retrieve(&"a");
        let third = cache.cache("a", "a");
        assert_eq!(first.unwrap(), "a");
        assert_eq!(second, None);
        assert_eq!(third, None);
    }

    #[test]
    fn multithreaded_test() {
        let writer = SimpleCache::new();
        let reader1 = writer.clone();
        let reader2 = writer.clone();
        let t0 = thread::spawn(move || {
            writer.cache("a", 0);
            writer.cache_ttl("b", 1, Duration::from_millis(20));
        });
        let t1 = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            assert_eq!(0, reader1.retrieve(&"a").unwrap());
            assert_eq!(1, reader1.retrieve(&"b").unwrap());
        });
        let t2 = thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            assert_eq!(0, reader2.retrieve(&"a").unwrap());
            assert_eq!(None, reader2.retrieve(&"b"));
        });

        t0.join().unwrap();
        t1.join().unwrap();
        t2.join().unwrap();
    }

    // This more of a sanity check than a benchmark
    #[test]
    fn single_threaded_10_000_000() {
        let cache = SimpleCache::new();
        for i in 0..10_000_000 {
            cache.cache(i, i);
        }
        let start = Instant::now();
        for j in 0..10_000_000 {
            assert_eq!(j, cache.retrieve(&j).unwrap());
        }
        let stop = Instant::now();
        dbg!((stop - start) / 10_000_000);
        assert!(stop < start.add(Duration::from_secs(10)));
    }

    // This more of a sanity check than a benchmark
    #[test]
    fn single_threaded_expired_10_000_000() {
        let cache = SimpleCache::new();
        for i in 0..10_000_000 {
            cache.cache_ttl(i, i, Duration::from_millis(10));
        }
        thread::sleep(Duration::from_millis(50));
        let start = Instant::now();
        for j in 0..10_000_000 {
            assert_eq!(None, cache.retrieve(&j));
        }
        let stop = Instant::now();
        dbg!((stop - start) / 10_000_000);
        assert!(stop < start.add(Duration::from_secs(20)));
    }
}
