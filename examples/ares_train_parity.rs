/*
Ares TRAINING-PATH parity gate.

This reproduces the EXACT representation the trainer feeds the net (see
examples/ares.rs `AresBinpackLoader`): for every corpus FEN, build
    ChessBoard::from_str("<fen> | 0 | 0.5")        // STM-relative encoding
    AresChessBoard { board, stm_is_black: false }  // literal training wrapper
then run `map_features` (the actual training feature path) and collect the
stm-index set and ntm-index set.

Compare to the engine ground truth in data/parity_corpus.txt:
    POV 0 = absolute White features
    POV 1 = absolute Black features
Mapped by ACTUAL side-to-move (parsed from the FEN):
    white-to-move: stm-set == POV0, ntm-set == POV1
    black-to-move: stm-set == POV1 (mover), ntm-set == POV0

Prints the pass count over all positions (644) and all sets (1288), and the
first few divergences (FEN, side-to-move, which set, symmetric diff).

Run:
    cargo run --release --example ares_train_parity -- ../../data/parity_corpus.txt
*/
use std::{collections::BTreeSet, fs, str::FromStr};

use bullet_lib::game::inputs::{AresChessBoard, AresThreats};
use bulletformat::ChessBoard;

fn parse_corpus(text: &str) -> Vec<(String, Vec<usize>, Vec<usize>)> {
    let mut out = Vec::new();
    let mut cur_fen: Option<String> = None;
    let mut pov0: Option<Vec<usize>> = None;
    let parse_list = |s: &str| -> Vec<usize> {
        let inner = s.trim().trim_start_matches('[').trim_end_matches(']');
        inner
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .collect()
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("FEN ") {
            cur_fen = Some(rest.trim().to_string());
            pov0 = None;
        } else if let Some(rest) = line.strip_prefix("POV 0 count ") {
            let list = rest.splitn(2, ':').nth(1).unwrap_or("");
            pov0 = Some(parse_list(list));
        } else if let Some(rest) = line.strip_prefix("POV 1 count ") {
            let list = rest.splitn(2, ':').nth(1).unwrap_or("");
            let p1 = parse_list(list);
            let fen = cur_fen.clone().expect("POV1 before FEN");
            let p0 = pov0.clone().expect("POV1 before POV0");
            out.push((fen, p0, p1));
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let in_path = args.get(1).map(String::as_str).unwrap_or("../../data/parity_corpus.txt");
    let text = fs::read_to_string(in_path).expect("read corpus");
    let corpus = parse_corpus(&text);
    eprintln!("loaded {} positions from {in_path}", corpus.len());

    let mut pos_pass = 0usize;
    let mut set_pass = 0usize;
    let mut total_sets = 0usize;
    let mut shown = 0usize;

    for (fen, pov0, pov1) in &corpus {
        let black_to_move = fen.split_whitespace().nth(1) == Some("b");

        // EXACT training representation.
        let entry = format!("{fen} | 0 | 0.5");
        let board = ChessBoard::from_str(&entry).unwrap_or_else(|e| panic!("bad FEN '{fen}': {e}"));
        let acb = AresChessBoard { board, stm_is_black: false };

        // TRAINING PATH: map_features (stm_is_black=false, as the loader does).
        let mut stm_set: BTreeSet<usize> = BTreeSet::new();
        let mut ntm_set: BTreeSet<usize> = BTreeSet::new();
        AresThreats::map_pairs(&acb).iter().for_each(|&(s, n)| {
            stm_set.insert(s);
            ntm_set.insert(n);
        });

        // Engine ground truth mapped by actual side-to-move.
        let (exp_stm, exp_ntm): (&Vec<usize>, &Vec<usize>) =
            if black_to_move { (pov1, pov0) } else { (pov0, pov1) };
        let exp_stm: BTreeSet<usize> = exp_stm.iter().copied().collect();
        let exp_ntm: BTreeSet<usize> = exp_ntm.iter().copied().collect();

        let stm_ok = stm_set == exp_stm;
        let ntm_ok = ntm_set == exp_ntm;
        total_sets += 2;
        if stm_ok {
            set_pass += 1;
        }
        if ntm_ok {
            set_pass += 1;
        }
        if stm_ok && ntm_ok {
            pos_pass += 1;
        } else if shown < 8 {
            shown += 1;
            let stmv: &str = if black_to_move { "black-to-move" } else { "white-to-move" };
            eprintln!("MISMATCH [{stmv}] {fen}");
            if !stm_ok {
                let missing: Vec<_> = exp_stm.difference(&stm_set).take(12).collect();
                let extra: Vec<_> = stm_set.difference(&exp_stm).take(12).collect();
                eprintln!(
                    "  STM set: have {} expect {} | missing(eng-only) {:?} extra(train-only) {:?}",
                    stm_set.len(),
                    exp_stm.len(),
                    missing,
                    extra
                );
            }
            if !ntm_ok {
                let missing: Vec<_> = exp_ntm.difference(&ntm_set).take(12).collect();
                let extra: Vec<_> = ntm_set.difference(&exp_ntm).take(12).collect();
                eprintln!(
                    "  NTM set: have {} expect {} | missing(eng-only) {:?} extra(train-only) {:?}",
                    ntm_set.len(),
                    exp_ntm.len(),
                    missing,
                    extra
                );
            }
        }
    }

    println!("TRAIN-PATH PARITY: positions {pos_pass}/{} sets {set_pass}/{total_sets}", corpus.len());
    if set_pass != total_sets {
        std::process::exit(1);
    }
}
