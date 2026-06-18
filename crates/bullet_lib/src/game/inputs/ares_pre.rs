//! Premapped Ares features — a compact, fixed-size POD record holding the
//! already-computed (stm, ntm) feature index pairs for a single training
//! position, plus its label (score/result) and output bucket.
//!
//! This is the fast-path counterpart to [`AresThreats`](super::AresThreats):
//! the threat features are computed ONCE by the `ares_precompute` tool and
//! written to a flat file of `AresPremapped` records, which bullet's native
//! `DirectSequentialDataLoader` mmaps and feeds directly. At train time
//! `map_features` is just a copy of the stored index pairs — NO threat
//! computation — so training becomes GPU-bound instead of feature-gen-bound.
//!
//! Feature pairing: every active Ares feature exists from both perspectives and
//! differs only in index; [`AresThreats::map_features`] emits exactly one
//! `(stm_idx, ntm_idx)` pair per active feature. We store those pairs 1:1, so a
//! single count `n` bounds both arrays.

use super::SparseInputType;
use crate::game::outputs::OutputBuckets;
use crate::value::loader::{CanBeDirectlySequentiallyLoaded, GameResult, LoadableDataType};

/// Fixed upper bound on active features per position. Measured corpus max is 84
/// (32 PST + threats); 160 leaves wide margin for dense positions. The precompute
/// tool clamps and counts any position exceeding this.
pub const NNZ: usize = 160;

/// Combined PST+threat input space size (PST-first; matches `AresThreats`).
const NUM_INPUTS: usize = 74544;

/// Compact POD training record: premapped feature index pairs + label + bucket.
///
/// `#[repr(C)]`, fixed size, no padding surprises. `unsafe impl
/// CanBeDirectlySequentiallyLoaded` asserts it is valid to transmute from any
/// byte sequence of its size — true here because every field is a plain integer
/// with no invalid bit patterns.
///
/// NOTE on index width: the combined PST+threat index space is [0, 74544),
/// which EXCEEDS `u16::MAX` (65535). Indices must therefore be stored as `u32`
/// (the task's `[u16; 160]` would silently truncate every threat index >= 65536
/// and is internally inconsistent with `num_inputs = 74544`). `u32` is the
/// minimal lossless width.
///
/// Layout (little-endian on disk; size = 1288 bytes):
///   stm: [u32; 160]  — stm-perspective feature indices, [0, 74544)
///   ntm: [u32; 160]  — ntm-perspective feature indices, paired 1:1 with stm
///   n:    u16        — number of active features (bounds the read of stm/ntm)
///   score:i16        — stm-relative centipawn score
///   result:u8        — stm-relative WDL: 0=loss, 1=draw, 2=win
///   bucket:u8        — output bucket (engine OUTPUT_BUCKETS_LAYOUT)
///   _pad: [u8; 10]   — explicit padding to a clean 1296-byte (16-aligned) size
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AresPremapped {
    pub stm: [u32; NNZ],
    pub ntm: [u32; NNZ],
    pub n: u16,
    pub score: i16,
    pub result: u8,
    pub bucket: u8,
    pub _pad: [u8; 10],
}

/// Record size is part of the on-disk format contract — lock it at compile time.
const _: () = assert!(std::mem::size_of::<AresPremapped>() == 1296);

impl Default for AresPremapped {
    fn default() -> Self {
        AresPremapped { stm: [0; NNZ], ntm: [0; NNZ], n: 0, score: 0, result: 1, bucket: 0, _pad: [0; 10] }
    }
}

// SAFETY: every field is a plain integer array/scalar; all bit patterns are
// valid, and the struct is `#[repr(C)]` + `Copy`. It is therefore sound to
// transmute it from any byte sequence of `size_of::<AresPremapped>()`.
unsafe impl CanBeDirectlySequentiallyLoaded for AresPremapped {}

// AresPremapped is plain POD; the derived Copy makes it Send + Sync already,
// but DataReader needs the bounds — they are satisfied automatically.

/// Fast-path Ares input: replays the precomputed `(stm, ntm)` index pairs.
///
/// Drop-in replacement for [`AresThreats`](super::AresThreats) on the trainer
/// side, with `RequiredDataType = AresPremapped`. `map_features` performs no
/// threat computation; it simply emits the stored pairs.
#[derive(Clone, Copy, Debug, Default)]
pub struct AresThreatsPre;

impl SparseInputType for AresThreatsPre {
    type RequiredDataType = AresPremapped;

    fn num_inputs(&self) -> usize {
        NUM_INPUTS
    }

    fn max_active(&self) -> usize {
        // Must be >= NNZ so the per-batch index buffer is large enough.
        256
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let n = pos.n as usize;
        for i in 0..n {
            f(pos.stm[i] as usize, pos.ntm[i] as usize);
        }
    }

    fn shorthand(&self) -> String {
        format!("{NUM_INPUTS}")
    }

    fn description(&self) -> String {
        "Ares combined PST + threat features, premapped (precomputed index pairs)".to_string()
    }
}

/// Trainer label plumbing: read the stored score and WDL result directly.
impl LoadableDataType for AresPremapped {
    fn score(&self) -> i16 {
        self.score
    }

    fn result(&self) -> GameResult {
        [GameResult::Loss, GameResult::Draw, GameResult::Win][self.result as usize]
    }
}

/// Output-bucket selector: the bucket is precomputed and stored in the record.
#[derive(Clone, Copy, Default)]
pub struct AresPreOutputBuckets;

impl OutputBuckets<AresPremapped> for AresPreOutputBuckets {
    const BUCKETS: usize = 8;

    fn bucket(&self, pos: &AresPremapped) -> u8 {
        pos.bucket
    }
}
