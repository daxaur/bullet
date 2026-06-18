/*
Ares parity harness (CPU-only, no GPU / no cuda feature required).

For each FEN in data/parity_corpus.txt, builds a `bulletformat::ChessBoard`,
reconstructs the absolute board, runs the AresThreats SparseInputType feature
computation, and prints the corpus format:

    FEN <fen>
    POV 0 count N : [sorted white(stm-pov0) indices]
    POV 1 count N : [sorted black(pov1) indices]

POV0 = White perspective, POV1 = Black perspective (matching tools/dump.rs).

Run:
    cargo run --release --example ares_parity -- \
        ../../data/parity_corpus.txt /tmp/ares_parity_out.txt
then:
    ./.venv/bin/python pipeline/check_parity.py /tmp/ares_parity_out.txt
*/
use std::{
    fs,
    io::{BufWriter, Write},
    str::FromStr,
};

use bullet_lib::game::inputs::{AresChessBoard, AresThreats};
use bulletformat::ChessBoard;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let in_path = args.get(1).map(String::as_str).unwrap_or("../../data/parity_corpus.txt");
    let out_path = args.get(2).map(String::as_str).unwrap_or("/tmp/ares_parity_out.txt");

    let text = fs::read_to_string(in_path).expect("read corpus");
    let mut fens: Vec<String> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("FEN ") {
            fens.push(rest.trim().to_string());
        }
    }
    eprintln!("loaded {} FENs from {in_path}", fens.len());

    let out = fs::File::create(out_path).expect("create out");
    let mut w = BufWriter::new(out);

    for fen in &fens {
        let stm_is_black = fen.split_whitespace().nth(1) == Some("b");

        // bulletformat parses "fen | score | wdl"
        let entry = format!("{fen} | 0 | 0.5");
        let board = ChessBoard::from_str(&entry)
            .unwrap_or_else(|e| panic!("bad FEN '{fen}': {e}"));

        let acb = AresChessBoard { board, stm_is_black };

        let pov0 = AresThreats::features(&acb, true); // White
        let pov1 = AresThreats::features(&acb, false); // Black

        writeln!(w, "FEN {fen}").unwrap();
        writeln!(w, "POV 0 count {} : {:?}", pov0.len(), pov0).unwrap();
        writeln!(w, "POV 1 count {} : {:?}", pov1.len(), pov1).unwrap();
    }

    w.flush().unwrap();
    eprintln!("wrote {out_path}");
}
