/*
Ares precompute tool (CPU-only, no GPU / no cuda feature).

Reads a Stockfish binpack via `SfBinpackLoader` using the SAME filter as
examples/ares.rs (ply>=16, not in check, |score|<=10000, normal move, not a
capture), computes the Ares PST+threat features for each position by REUSING
the existing `AresThreats` logic in the TRAINING relative frame
(`AresChessBoard { board, stm_is_black: false }` — exactly as ares.rs's
`AresBinpackLoader`), and writes compact fixed-size `AresPremapped` records to
an output `.aresdata` file. At train time these are loaded directly with
bullet's native `DirectSequentialDataLoader` so `map_features` is just a copy.

Usage (precompute mode):
    cargo run --release --example ares_precompute -- <input.binpack> <output.aresdata> [max_positions]
  or via env:
    ARES_BINPACK=in.binpack ARES_DATA=out.aresdata ARES_MAX_POS=1000000 \
        cargo run --release --example ares_precompute

Roundtrip / feature-preservation gate (mode "roundtrip"):
    cargo run --release --example ares_precompute -- roundtrip <corpus.txt>
  For the corpus FENs, computes features two ways — directly via AresThreats
  map_features (relative frame, stm_is_black=false) AND via
  precompute->write->read-back AresPremapped->AresThreatsPre map_features — and
  asserts the emitted (stm,ntm) index multisets are IDENTICAL per position.
  Prints "ROUNDTRIP OK" + count; exits nonzero on any mismatch.
*/
use std::{
    fs,
    io::{BufWriter, Write},
    str::FromStr,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use bullet_lib::{
    game::inputs::{
        ARES_NNZ, AresChessBoard, AresPremapped, AresThreats, AresThreatsPre, OUTPUT_BUCKETS_LAYOUT,
    },
    game::inputs::SparseInputType,
    value::loader::{
        CanBeDirectlySequentiallyLoaded, DataReader, DirectSequentialDataLoader, GameResult, LoadableDataType,
        sfbinpack::{MoveType, PieceType, SfBinpackLoader, TrainingDataEntry},
    },
};
use bulletformat::{BulletFormat, ChessBoard};

fn filter(e: &TrainingDataEntry) -> bool {
    e.ply >= 16
        && !e.pos.is_checked(e.pos.side_to_move())
        && e.score.unsigned_abs() <= 10000
        && e.mv.mtype() == MoveType::Normal
        && e.pos.piece_at(e.mv.to()).piece_type() == PieceType::None
}

/// Build an `AresPremapped` from a single STM-relative bulletformat `ChessBoard`,
/// reusing the existing `AresThreats` feature logic. Returns the record plus a
/// flag indicating whether the feature count was clamped (n > NNZ).
fn premap(board: &ChessBoard) -> (AresPremapped, bool) {
    let acb = AresChessBoard { board: *board, stm_is_black: false };
    let pairs = AresThreats::map_pairs(&acb);

    let mut rec = AresPremapped::default();
    let mut clamped = false;
    let mut n = pairs.len();
    if n > ARES_NNZ {
        clamped = true;
        n = ARES_NNZ;
    }
    for (i, &(s, t)) in pairs.iter().take(n).enumerate() {
        rec.stm[i] = s as u32;
        rec.ntm[i] = t as u32;
    }
    rec.n = n as u16;

    // Labels: score is stm-relative (binpack board already STM-relative).
    rec.score = BulletFormat::score(board);
    rec.result = match acb.result() {
        GameResult::Loss => 0,
        GameResult::Draw => 1,
        GameResult::Win => 2,
    };
    rec.bucket = OUTPUT_BUCKETS_LAYOUT[board.occ().count_ones() as usize];

    (rec, clamped)
}

fn run_precompute(binpack: &str, out_path: &str, max_pos: usize) {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    eprintln!("precompute: {binpack} -> {out_path} (max_pos={max_pos}, threads={threads})");

    let out = fs::File::create(out_path).expect("create output");
    let writer = Mutex::new(BufWriter::new(out));
    let written = AtomicU64::new(0);
    let clamped_count = AtomicU64::new(0);
    let done = AtomicU64::new(0); // 1 once we've hit max_pos

    let loader = SfBinpackLoader::new(binpack, 1024, threads, filter);

    loader.read_chunks(0, |chunk: &[ChessBoard]| {
        if done.load(Ordering::Relaxed) != 0 {
            return false;
        }

        // Parallel map over the chunk, then a single ordered write.
        let recs: Vec<(AresPremapped, bool)> =
            std::thread::scope(|scope| {
                let nthreads = threads.max(1);
                let part = chunk.len().div_ceil(nthreads);
                let mut handles = Vec::new();
                for t in 0..nthreads {
                    let lo = t * part;
                    let hi = (lo + part).min(chunk.len());
                    if lo >= hi {
                        break;
                    }
                    let slice = &chunk[lo..hi];
                    handles.push(scope.spawn(move || slice.iter().map(premap).collect::<Vec<_>>()));
                }
                let mut all = Vec::with_capacity(chunk.len());
                for h in handles {
                    all.extend(h.join().unwrap());
                }
                all
            });

        let mut w = writer.lock().unwrap();
        let mut local_clamped = 0u64;
        for (rec, clamped) in &recs {
            if written.load(Ordering::Relaxed) as usize >= max_pos {
                done.store(1, Ordering::Relaxed);
                break;
            }
            if *clamped {
                local_clamped += 1;
            }
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    (rec as *const AresPremapped).cast::<u8>(),
                    std::mem::size_of::<AresPremapped>(),
                )
            };
            w.write_all(bytes).expect("write record");
            written.fetch_add(1, Ordering::Relaxed);
        }
        clamped_count.fetch_add(local_clamped, Ordering::Relaxed);

        let n = written.load(Ordering::Relaxed);
        if n % 1_000_000 < chunk.len() as u64 {
            eprintln!("  written {n} positions...");
        }

        done.load(Ordering::Relaxed) == 0
    });

    writer.lock().unwrap().flush().expect("flush");
    let total = written.load(Ordering::Relaxed);
    let clamped = clamped_count.load(Ordering::Relaxed);
    eprintln!(
        "DONE: wrote {total} positions ({} bytes/record). clamped (n>{ARES_NNZ}): {clamped}",
        std::mem::size_of::<AresPremapped>()
    );
}

/// Feature-preservation gate: for every corpus FEN, compare the (stm,ntm)
/// multiset from direct AresThreats map_features vs. the
/// precompute->write->read-back->AresThreatsPre map_features path.
fn run_roundtrip(corpus: &str) {
    // Confirm the record really satisfies the direct-load trait at compile time.
    fn assert_pod<T: CanBeDirectlySequentiallyLoaded>() {}
    assert_pod::<AresPremapped>();

    let text = fs::read_to_string(corpus).expect("read corpus");
    let mut fens: Vec<String> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("FEN ") {
            fens.push(rest.trim().to_string());
        }
    }
    eprintln!("roundtrip: {} FENs from {corpus}", fens.len());

    // Build direct reference + premapped records.
    let mut direct: Vec<Vec<(usize, usize)>> = Vec::with_capacity(fens.len());
    let mut records: Vec<AresPremapped> = Vec::with_capacity(fens.len());
    for fen in &fens {
        let entry = format!("{fen} | 0 | 0.5");
        let board = ChessBoard::from_str(&entry).unwrap_or_else(|e| panic!("bad FEN '{fen}': {e}"));
        let acb = AresChessBoard { board, stm_is_black: false };

        let mut d = AresThreats::map_pairs(&acb);
        d.sort_unstable();
        direct.push(d);

        let (rec, _clamped) = premap(&board);
        records.push(rec);
    }

    // Write -> read back through the actual on-disk format + native loader.
    let tmp = std::env::temp_dir().join("ares_roundtrip.aresdata");
    let tmp_path = tmp.to_str().unwrap().to_string();
    {
        let f = fs::File::create(&tmp_path).expect("create tmp");
        let mut w = BufWriter::new(f);
        for rec in &records {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    (rec as *const AresPremapped).cast::<u8>(),
                    std::mem::size_of::<AresPremapped>(),
                )
            };
            w.write_all(bytes).unwrap();
        }
        w.flush().unwrap();
    }

    let loader = DirectSequentialDataLoader::new(&[&tmp_path]);
    let mut readback: Vec<Vec<(usize, usize)>> = Vec::with_capacity(fens.len());
    DataReader::<AresPremapped>::read_chunks(&loader, 0, |chunk: &[AresPremapped]| {
        for rec in chunk {
            let mut pairs = Vec::new();
            AresThreatsPre.map_features(rec, |s, n| pairs.push((s, n)));
            pairs.sort_unstable();
            readback.push(pairs);
        }
        true
    });

    assert_eq!(direct.len(), readback.len(), "count mismatch direct={} readback={}", direct.len(), readback.len());

    let mut mismatches = 0usize;
    for (i, (d, r)) in direct.iter().zip(readback.iter()).enumerate() {
        if d != r {
            mismatches += 1;
            eprintln!("MISMATCH at #{i} ({}): direct={} readback={} pairs", i, d.len(), r.len());
        }
    }

    if mismatches != 0 {
        eprintln!("ROUNDTRIP FAILED: {mismatches}/{} positions mismatched", direct.len());
        std::process::exit(1);
    }
    println!("ROUNDTRIP OK {} positions", direct.len());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.get(1).map(String::as_str) == Some("roundtrip") {
        let corpus = args.get(2).map(String::as_str).unwrap_or("../../data/parity_corpus.txt");
        run_roundtrip(corpus);
        return;
    }

    let binpack = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("ARES_BINPACK").ok())
        .unwrap_or_else(|| "data/ares.binpack".to_string());
    let out_path = args
        .get(2)
        .cloned()
        .or_else(|| std::env::var("ARES_DATA").ok())
        .unwrap_or_else(|| "data/ares.aresdata".to_string());
    let max_pos = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .or_else(|| std::env::var("ARES_MAX_POS").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(usize::MAX);

    run_precompute(&binpack, &out_path, max_pos);
}
