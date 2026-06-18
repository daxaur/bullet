/*
Ares NNUE trainer — the REAL engine architecture (eval-parity target).

Arch (matches the Reckless/Ares engine bit-for-bit, verified against engine source):
  - Input: AresThreats SparseInputType, num_inputs = 74544
        (PST-first [0, 7680); threats [7680, 74544)).
  - Single FT affine l0: 74544 -> 768 per perspective.
  - FT activation = CReLU then PAIRWISE-MULTIPLY (SCReLU-pairwise), 768 -> 384:
        ft(x, s, e) = l0.slice(s, e).forward(x).crelu();
        stm_hidden  = ft(stm, 0, 384) * ft(stm, 384, 768);   // 384
        ntm_hidden  = ft(ntm, 0, 384) * ft(ntm, 384, 768);   // 384
        hl1 = stm_hidden.concat(ntm_hidden);                 // 768
  - Bucketed head (8 output buckets), 768 -> 16 -> 32 -> 1, CReLU activations
        (engine clamps 0..1 == crelu; verified in forward/scalar.rs):
        l1: 768 -> 8*16,  l2: 16 -> 8*32,  l3: 32 -> 8.
  - eval_scale = 380.0 (engine NETWORK_SCALE; NOT 400).
  - loss = output.sigmoid().squared_error(target)  (MSE baseline).
  - dual_perspective, AdamW.

Output buckets: AresOutputBuckets (engine OUTPUT_BUCKETS_LAYOUT keyed by occupancy
popcount). This is NOT MaterialCount — that diverges from the engine table in 13/31
buckets.

Data: real Stockfish binpacks via SfBinpackLoader (same filter as ares_lite.rs:
ply>=16, not in check, |score|<=10000, normal move, not a capture). SfBinpackLoader
yields bulletformat `ChessBoard` but AresThreats::RequiredDataType is `AresChessBoard`;
we wrap the loader in a thin adapter (`AresBinpackLoader`) that maps each ChessBoard to
`AresChessBoard { board, stm_is_black: false }`. The binpack ChessBoard is already
STM-relative (convert_to_bulletformat negates score/result for black STM and stores the
board side-relative), so stm_is_black = false makes the reconstructed "absolute" frame
equal the STM-relative frame — i.e. features are computed STM-as-white-relative, which is
exactly the training frame we want. (Absolute white/black framing is only needed by the
parity gate, which feeds known-STM FENs directly.)

SAVE FORMAT: RAW f32 tensors (NO .quantise), named l0w,l0b,l1w,l1b,l2w,l2b,l3w,l3b.
Our separate Python converter (pipeline/convert_net.py) does quantization + remap.

Run on a CUDA/ROCm GPU (free Kaggle session is enough):
    # place a binpack at the BINPACK_PATH below, then:
    cargo run --release --example ares --features cuda
Local CPU gate (no GPU): `cargo check --example ares` compiles without the cuda feature.
*/
use bullet_lib::{
    game::inputs::{AresChessBoard, AresOutputBuckets, AresThreats},
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{self, DataReader, sfbinpack::TrainingDataEntry},
    },
};
use bulletformat::ChessBoard;

// ───────────────────────── clearly-marked knobs ─────────────────────────

/// Number of superbatches to train. Tune for a real run on GPU.
const SUPERBATCHES: usize = 240;
/// Path to the Stockfish binpack training data (set before a real GPU run).
const BINPACK_PATH: &str = "data/ares.binpack";
/// Net identifier used for checkpoint naming.
const NET_ID: &str = "ares-net";

const FT_SIZE: usize = 768; // per-perspective FT width (CReLU), pairwise -> 384
const L1_OUT: usize = 16; // first head hidden width
const L2_OUT: usize = 32; // second head hidden width
const NUM_OUTPUT_BUCKETS: usize = 8;

// ───────────────────── loader adapter: ChessBoard -> AresChessBoard ─────────────────────

/// Wraps an `SfBinpackLoader` (which yields bulletformat `ChessBoard`) and presents a
/// `DataReader<AresChessBoard>` by mapping each board to `AresChessBoard`. The binpack
/// board is STM-relative, so `stm_is_black: false` keeps the training frame
/// STM-as-white-relative.
#[derive(Clone)]
struct AresBinpackLoader<L>(L);

impl<L> DataReader<AresChessBoard> for AresBinpackLoader<L>
where
    L: DataReader<ChessBoard>,
{
    fn read_chunks<F: FnMut(&[AresChessBoard]) -> bool>(&self, skip_count: usize, mut f: F) {
        self.0.read_chunks(skip_count, |chunk: &[ChessBoard]| {
            let mapped: Vec<AresChessBoard> =
                chunk.iter().map(|&board| AresChessBoard { board, stm_is_black: false }).collect();
            f(&mapped)
        });
    }
}

fn main() {
    let initial_lr = 0.001;
    let final_lr = 0.001 * 0.3f32.powi(5);
    let wdl_proportion = 0.75;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(AresThreats)
        .output_buckets(AresOutputBuckets)
        // RAW f32 tensors — no quantise, no transpose. Python convert_net.py handles
        // quantization + layout remap to the engine net format.
        .save_format(&[
            SavedFormat::id("l0w"),
            SavedFormat::id("l0b"),
            SavedFormat::id("l1w"),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w"),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w"),
            SavedFormat::id("l3b"),
        ])
        // MSE baseline (the loss we will A/B HL-Gauss against next).
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", 74544, FT_SIZE);
            let l1 = builder.new_affine("l1", 2 * (FT_SIZE / 2), NUM_OUTPUT_BUCKETS * L1_OUT);
            let l2 = builder.new_affine("l2", L1_OUT, NUM_OUTPUT_BUCKETS * L2_OUT);
            let l3 = builder.new_affine("l3", L2_OUT, NUM_OUTPUT_BUCKETS);

            // FT: CReLU then pairwise-multiply (SCReLU-pairwise), 768 -> 384 per perspective.
            let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
            let stm_hidden = ft(stm_inputs, 0, FT_SIZE / 2) * ft(stm_inputs, FT_SIZE / 2, FT_SIZE);
            let ntm_hidden = ft(ntm_inputs, 0, FT_SIZE / 2) * ft(ntm_inputs, FT_SIZE / 2, FT_SIZE);

            let hl1 = stm_hidden.concat(ntm_hidden); // 768

            // Bucketed head with CReLU activations (engine clamps 0..1 = crelu).
            let hl2 = l1.forward(hl1).select(output_buckets).crelu();
            let hl3 = l2.forward(hl2).select(output_buckets).crelu();
            l3.forward(hl3).select(output_buckets)
        });

    // Env overrides so a cheap VALIDATION run (few superbatches) can produce a checkpoint to
    // exercise the converter + eval-parity before committing to a full ~4h GPU run.
    let env_usize = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let superbatches = env_usize("ARES_SUPERBATCHES", SUPERBATCHES);
    let save_rate = env_usize("ARES_SAVE_RATE", 40).min(superbatches.max(1));
    let binpack = std::env::var("ARES_BINPACK").unwrap_or_else(|_| BINPACK_PATH.to_string());

    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: 380.0, // engine NETWORK_SCALE (NOT 400)
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_proportion },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate,
    };

    let settings =
        LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    // Real Stockfish binpack data, same filter as ares_lite.rs, wrapped to yield AresChessBoard.
    let data_loader = {
        use loader::sfbinpack::{MoveType, PieceType, SfBinpackLoader};
        fn filter(e: &TrainingDataEntry) -> bool {
            e.ply >= 16
                && !e.pos.is_checked(e.pos.side_to_move())
                && e.score.unsigned_abs() <= 10000
                && e.mv.mtype() == MoveType::Normal
                && e.pos.piece_at(e.mv.to()).piece_type() == PieceType::None
        }
        AresBinpackLoader(SfBinpackLoader::new(&binpack, 1024, 4, filter))
    };

    trainer.run(&schedule, &settings, &data_loader);
}
