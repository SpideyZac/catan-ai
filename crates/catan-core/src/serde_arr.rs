//! Serde support for fixed-size arrays longer than 32 elements.

use serde::de::Error;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub fn serialize<S: Serializer, T: Serialize, const N: usize>(a: &[T; N], s: S) -> Result<S::Ok, S::Error> {
    a.as_slice().serialize(s)
}

pub fn deserialize<'de, D: Deserializer<'de>, T: Deserialize<'de>, const N: usize>(d: D) -> Result<[T; N], D::Error> {
    let v: Vec<T> = Vec::deserialize(d)?;
    let len = v.len();
    v.try_into()
        .map_err(|_| D::Error::custom(format!("expected array of length {N}, got {len}")))
}
