/*
Ares NNUE trainer — PRECOMPUTED fast-path variant of examples/ares.rs.

Identical architecture / loss / scale / output buckets / save format to ares.rs,
but the data path is GPU-bound instead of feature-gen-bound:

  - Inputs:  `AresThreatsPre` (RequiredDataType = `AresPremapped`). `map_features`
             is a COPY of the precomputed (stm, ntm) index pairs — no threat
             computation at train time.
  - Buckets: `AresPreOutputBuckets` (reads the bucket stored in the record).
  - Data:    a flat `.aresdata` file of `AresPremapped` records produced by
             `examples/ares_precompute.rs`, loaded with bullet's native
             `DirectSequentialDataLoader` (mmap + zero-copy transmute).

Generate the data first (CPU):
    cargo run --release --example ares_precompute -- in.binpack data.aresdata 100000000
Then train (GPU):
    ARES_DATA=data.aresdata cargo run --release --example ares_pre --features cuda

Env knobs: ARES_SUPERBATCHES, ARES_SAVE_RATE, ARES_DATA (path to the .aresdata).
Local CPU gate (no GPU): `cargo check --example ares_pre` compiles without cuda.
*/
use bullet_lib::{
    game::inputs::{AresPreOutputBuckets, AresThreatsPre},
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

// ───────────────────────── clearly-marked knobs ─────────────────────────

/// Number of superbatches to train. Tune for a real run on GPU.
const SUPERBATCHES: usize = 240;
/// Path to the precomputed `.aresdata` file (set before a real GPU run).
const DATA_PATH: &str = "data/ares.aresdata";
/// Net identifier used for checkpoint naming.
const NET_ID: &str = "ares-net";

const FT_SIZE: usize = 768; // per-perspective FT width (CReLU), pairwise -> 384
const L1_OUT: usize = 16; // first head hidden width
const L2_OUT: usize = 32; // second head hidden width
const NUM_OUTPUT_BUCKETS: usize = 8;

fn main() {
    let initial_lr = 0.001;
    let final_lr = 0.001 * 0.3f32.powi(5);
    let wdl_proportion = 0.75;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(AresThreatsPre)
        .output_buckets(AresPreOutputBuckets)
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

    let env_usize = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let superbatches = env_usize("ARES_SUPERBATCHES", SUPERBATCHES);
    let save_rate = env_usize("ARES_SAVE_RATE", 40).min(superbatches.max(1));
    let data = std::env::var("ARES_DATA").unwrap_or_else(|_| DATA_PATH.to_string());

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

    // Precomputed premapped records — native direct loader (map_features is a copy).
    let data_loader = DirectSequentialDataLoader::new(&[data.as_str()]);

    trainer.run(&schedule, &settings, &data_loader);
}
