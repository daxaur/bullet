//! Ares combined NNUE features (PST + threats) computed on-the-fly from a
//! `bulletformat::ChessBoard`, matching the Reckless engine bit-for-bit.
//!
//! This is a self-contained port of the minimal engine modules needed to
//! reproduce the corpus oracle `tools/dump.rs::features_for_pov`. Sources copied
//! / adapted from the engine tree (`engine/Reckless`):
//!   - src/tools/dump.rs                              (features_for_pov / pst_index oracle)
//!   - src/nnue.rs                                    (INPUT_BUCKETS=10, INPUT_BUCKETS_LAYOUT)
//!   - src/nnue/accumulator/threats/threat_index.rs   (threat_index + initialize)
//!   - src/nnue/accumulator/psq.rs                    (pst_index)
//!   - src/lookup.rs + build/{attacks,maps,magics}.rs (magic-bitboard attack tables)
//!   - src/types/{square,piece,color,bitboard}.rs     (Square/Color/Piece/PieceType/Bitboard)
//!
//! Encoding (PST-FIRST): PST indices in [0, 7680); threat indices offset by
//! THREAT_BASE = 7680 into [7680, 74544). num_inputs = 74544.
//!
//! IMPORTANT — coordinate frame: a `bulletformat::ChessBoard` is stored
//! side-to-move-relative (when the original side to move was Black, the board is
//! vertically flipped and colors are swapped during FEN parsing). The engine's
//! corpus oracle, however, dumps features in ABSOLUTE white/black terms
//! (POV 0 = White perspective, POV 1 = Black perspective, regardless of STM).
//! The harness therefore reconstructs the ABSOLUTE board (it knows the FEN's STM)
//! and exposes it here; see `AresBoard::from_absolute`. `map_features` emits
//! f(white_index, black_index) per active feature so POV0=white, POV1=black.

use std::sync::OnceLock;

use bulletformat::{BulletFormat, ChessBoard};

use super::SparseInputType;
use crate::game::outputs::OutputBuckets;
use crate::value::loader::{GameResult, LoadableDataType};

// ───────────────────────────── constants (nnue.rs) ─────────────────────────────

pub const INPUT_BUCKETS: usize = 10;
const PST_SIZE: usize = INPUT_BUCKETS * 768; // 7680
const THREAT_BASE: usize = PST_SIZE;
const NUM_INPUTS: usize = 74544;

#[rustfmt::skip]
pub const INPUT_BUCKETS_LAYOUT: [u8; 64] = [
    0, 1, 2, 3, 3, 2, 1, 0,
    4, 5, 6, 7, 7, 6, 5, 4,
    8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
];

// ───────────────────────────── types (types/*.rs) ─────────────────────────────

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Color {
    White = 0,
    Black = 1,
}
impl std::ops::Not for Color {
    type Output = Color;
    fn not(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}
impl PieceType {
    pub const NUM: usize = 6;
    const fn new(v: usize) -> Self {
        match v {
            0 => PieceType::Pawn,
            1 => PieceType::Knight,
            2 => PieceType::Bishop,
            3 => PieceType::Rook,
            4 => PieceType::Queen,
            _ => PieceType::King,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Piece(pub u8); // 0..12, layout = (piece_type << 1) | color  (matches engine Piece enum order)
impl Piece {
    pub const NUM: usize = 12;
    const fn new(color: Color, pt: PieceType) -> Self {
        Piece(((pt as u8) << 1) | color as u8)
    }
    const fn from_index(i: usize) -> Self {
        Piece(i as u8)
    }
    const ALL: [Piece; 12] = [
        Piece(0), Piece(1), Piece(2), Piece(3), Piece(4), Piece(5),
        Piece(6), Piece(7), Piece(8), Piece(9), Piece(10), Piece(11),
    ];
    const fn color(self) -> Color {
        if self.0 & 1 == 0 { Color::White } else { Color::Black }
    }
    const fn piece_type(self) -> PieceType {
        PieceType::new((self.0 >> 1) as usize)
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Square(pub u8); // 0..64, LERF
impl Square {
    const fn new(v: u8) -> Self {
        Square(v)
    }
    const fn file(self) -> u8 {
        self.0 & 7
    }
    fn is_kingside(self) -> bool {
        self.file() >= 4
    }
    const fn relative_to(self, c: Color) -> Self {
        match c {
            Color::White => self,
            Color::Black => Square(self.0 ^ 56),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Bitboard(pub u64);
impl Bitboard {
    const fn popcount(self) -> u32 {
        self.0.count_ones()
    }
}
impl Iterator for Bitboard {
    type Item = Square;
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            let lsb = self.0.trailing_zeros() as u8;
            self.0 &= self.0 - 1;
            Some(Square(lsb))
        }
    }
}
impl std::ops::BitAnd for Bitboard {
    type Output = Bitboard;
    fn bitand(self, r: Bitboard) -> Bitboard {
        Bitboard(self.0 & r.0)
    }
}

// ───────────────────── attack tables (build/{attacks,maps,magics}.rs) ─────────────────────

#[derive(Debug)]
pub struct MagicEntry {
    pub mask: u64,
    pub magic: u64,
    pub shift: u32,
    pub offset: usize,
}

pub static ROOK_MAGICS: [MagicEntry; 64] = [
    MagicEntry { mask: 0x000101010101017E, magic: 0x1080018022704002, shift: 52, offset: 0 },
    MagicEntry { mask: 0x000202020202027C, magic: 0x8B4000A005900840, shift: 53, offset: 4096 },
    MagicEntry { mask: 0x000404040404047A, magic: 0x09000A4100200012, shift: 53, offset: 6144 },
    MagicEntry { mask: 0x0008080808080876, magic: 0x0200100408402200, shift: 53, offset: 8192 },
    MagicEntry { mask: 0x001010101010106E, magic: 0x0600020020083064, shift: 53, offset: 10240 },
    MagicEntry { mask: 0x002020202020205E, magic: 0x0100020100040008, shift: 53, offset: 12288 },
    MagicEntry { mask: 0x004040404040403E, magic: 0x0600010450881200, shift: 53, offset: 14336 },
    MagicEntry { mask: 0x008080808080807E, magic: 0x8100028208412100, shift: 52, offset: 16384 },
    MagicEntry { mask: 0x0001010101017E00, magic: 0x2001800040008021, shift: 53, offset: 20480 },
    MagicEntry { mask: 0x0002020202027C00, magic: 0x0000C02010004001, shift: 54, offset: 22528 },
    MagicEntry { mask: 0x0004040404047A00, magic: 0x0001002000401104, shift: 54, offset: 23552 },
    MagicEntry { mask: 0x0008080808087600, magic: 0x1201002008100104, shift: 54, offset: 24576 },
    MagicEntry { mask: 0x0010101010106E00, magic: 0x0904808008000400, shift: 54, offset: 25600 },
    MagicEntry { mask: 0x0020202020205E00, magic: 0x4004800400020080, shift: 54, offset: 26624 },
    MagicEntry { mask: 0x0040404040403E00, magic: 0x040A0002000801C4, shift: 54, offset: 27648 },
    MagicEntry { mask: 0x0080808080807E00, magic: 0x1D020001084408A2, shift: 53, offset: 28672 },
    MagicEntry { mask: 0x00010101017E0100, magic: 0x41400080088A4620, shift: 53, offset: 30720 },
    MagicEntry { mask: 0x00020202027C0200, magic: 0x0C50004000200040, shift: 54, offset: 32768 },
    MagicEntry { mask: 0x00040404047A0400, magic: 0x0042020010804020, shift: 54, offset: 33792 },
    MagicEntry { mask: 0x0008080808760800, magic: 0x0400210008100104, shift: 54, offset: 34816 },
    MagicEntry { mask: 0x00101010106E1000, magic: 0x0828808004000800, shift: 54, offset: 35840 },
    MagicEntry { mask: 0x00202020205E2000, magic: 0x0600808002000400, shift: 54, offset: 36864 },
    MagicEntry { mask: 0x00404040403E4000, magic: 0x42010C0010090208, shift: 54, offset: 37888 },
    MagicEntry { mask: 0x00808080807E8000, magic: 0x0208020000408924, shift: 53, offset: 38912 },
    MagicEntry { mask: 0x000101017E010100, magic: 0x0040400080009020, shift: 53, offset: 40960 },
    MagicEntry { mask: 0x000202027C020200, magic: 0x1040008080200042, shift: 54, offset: 43008 },
    MagicEntry { mask: 0x000404047A040400, magic: 0x000A820200122240, shift: 54, offset: 44032 },
    MagicEntry { mask: 0x0008080876080800, magic: 0x0010040040080040, shift: 54, offset: 45056 },
    MagicEntry { mask: 0x001010106E101000, magic: 0x0208008080040008, shift: 54, offset: 46080 },
    MagicEntry { mask: 0x002020205E202000, magic: 0x5800200801041040, shift: 54, offset: 47104 },
    MagicEntry { mask: 0x004040403E404000, magic: 0x0806004200410804, shift: 54, offset: 48128 },
    MagicEntry { mask: 0x008080807E808000, magic: 0x4812004200041081, shift: 53, offset: 49152 },
    MagicEntry { mask: 0x0001017E01010100, magic: 0x2040204000800080, shift: 53, offset: 51200 },
    MagicEntry { mask: 0x0002027C02020200, magic: 0x20400C4081002501, shift: 54, offset: 53248 },
    MagicEntry { mask: 0x0004047A04040400, magic: 0x0010002000801881, shift: 54, offset: 54272 },
    MagicEntry { mask: 0x0008087608080800, magic: 0x1000801000800800, shift: 54, offset: 55296 },
    MagicEntry { mask: 0x0010106E10101000, magic: 0x8004800800800401, shift: 54, offset: 56320 },
    MagicEntry { mask: 0x0020205E20202000, magic: 0x1800800400800200, shift: 54, offset: 57344 },
    MagicEntry { mask: 0x0040403E40404000, magic: 0x8219000401000200, shift: 54, offset: 58368 },
    MagicEntry { mask: 0x0080807E80808000, magic: 0x000008410200008C, shift: 53, offset: 59392 },
    MagicEntry { mask: 0x00017E0101010100, magic: 0x0640802840008000, shift: 53, offset: 61440 },
    MagicEntry { mask: 0x00027C0202020200, magic: 0x3000201000404000, shift: 54, offset: 63488 },
    MagicEntry { mask: 0x00047A0404040400, magic: 0x9020220088420011, shift: 54, offset: 64512 },
    MagicEntry { mask: 0x0008760808080800, magic: 0x20920041110A0020, shift: 54, offset: 65536 },
    MagicEntry { mask: 0x00106E1010101000, magic: 0x0A04000800808006, shift: 54, offset: 66560 },
    MagicEntry { mask: 0x00205E2020202000, magic: 0x8400020004008080, shift: 54, offset: 67584 },
    MagicEntry { mask: 0x00403E4040404000, magic: 0x0089081002040081, shift: 54, offset: 68608 },
    MagicEntry { mask: 0x00807E8080808000, magic: 0x0401008044020001, shift: 53, offset: 69632 },
    MagicEntry { mask: 0x007E010101010100, magic: 0x0400800040002080, shift: 53, offset: 71680 },
    MagicEntry { mask: 0x007C020202020200, magic: 0x82400042211A8100, shift: 54, offset: 73728 },
    MagicEntry { mask: 0x007A040404040400, magic: 0x0410008010200080, shift: 54, offset: 74752 },
    MagicEntry { mask: 0x0076080808080800, magic: 0x3080080010008080, shift: 54, offset: 75776 },
    MagicEntry { mask: 0x006E101010101000, magic: 0x0080080004008080, shift: 54, offset: 76800 },
    MagicEntry { mask: 0x005E202020202000, magic: 0x0282020080040080, shift: 54, offset: 77824 },
    MagicEntry { mask: 0x003E404040404000, magic: 0x0000100108820400, shift: 54, offset: 78848 },
    MagicEntry { mask: 0x007E808080808000, magic: 0x0091340080410200, shift: 53, offset: 79872 },
    MagicEntry { mask: 0x7E01010101010100, magic: 0x0082800020430013, shift: 52, offset: 81920 },
    MagicEntry { mask: 0x7C02020202020200, magic: 0x0208408012002102, shift: 53, offset: 86016 },
    MagicEntry { mask: 0x7A04040404040400, magic: 0x0464220008401282, shift: 53, offset: 88064 },
    MagicEntry { mask: 0x7608080808080800, magic: 0x2100041000082101, shift: 53, offset: 90112 },
    MagicEntry { mask: 0x6E10101010101000, magic: 0x0402000409605002, shift: 53, offset: 92160 },
    MagicEntry { mask: 0x5E20202020202000, magic: 0x2005004882040001, shift: 53, offset: 94208 },
    MagicEntry { mask: 0x3E40404040404000, magic: 0x1040409008410204, shift: 53, offset: 96256 },
    MagicEntry { mask: 0x7E80808080808000, magic: 0x0800408849040022, shift: 52, offset: 98304 },
];

pub static BISHOP_MAGICS: [MagicEntry; 64] = [
    MagicEntry { mask: 0x0040201008040200, magic: 0x4041620809010010, shift: 58, offset: 0 },
    MagicEntry { mask: 0x0000402010080400, magic: 0x0010500101202102, shift: 59, offset: 64 },
    MagicEntry { mask: 0x0000004020100A00, magic: 0x024202004108001A, shift: 59, offset: 96 },
    MagicEntry { mask: 0x0000000040221400, magic: 0x0108260040011000, shift: 59, offset: 128 },
    MagicEntry { mask: 0x0000000002442800, magic: 0x0401104000000000, shift: 59, offset: 160 },
    MagicEntry { mask: 0x0000000204085000, magic: 0x0822011008062000, shift: 59, offset: 192 },
    MagicEntry { mask: 0x0000020408102000, magic: 0x10311808020A0002, shift: 59, offset: 224 },
    MagicEntry { mask: 0x0002040810204000, magic: 0x0030420201014100, shift: 58, offset: 256 },
    MagicEntry { mask: 0x0020100804020000, magic: 0x0852041082280110, shift: 59, offset: 320 },
    MagicEntry { mask: 0x0040201008040000, magic: 0x0000A00214011020, shift: 59, offset: 352 },
    MagicEntry { mask: 0x00004020100A0000, magic: 0x084012241400A450, shift: 59, offset: 384 },
    MagicEntry { mask: 0x0000004022140000, magic: 0x1300822082000805, shift: 59, offset: 416 },
    MagicEntry { mask: 0x0000000244280000, magic: 0x01241C10284008C0, shift: 59, offset: 448 },
    MagicEntry { mask: 0x0000020408500000, magic: 0x00100A2804C02224, shift: 59, offset: 480 },
    MagicEntry { mask: 0x0002040810200000, magic: 0x00040202411C40A0, shift: 59, offset: 512 },
    MagicEntry { mask: 0x0004081020400000, magic: 0x0006620202010410, shift: 59, offset: 544 },
    MagicEntry { mask: 0x0010080402000200, magic: 0x201080A020429090, shift: 59, offset: 576 },
    MagicEntry { mask: 0x0020100804000400, magic: 0x0015010208280300, shift: 59, offset: 608 },
    MagicEntry { mask: 0x004020100A000A00, magic: 0x0422040102020202, shift: 57, offset: 640 },
    MagicEntry { mask: 0x0000402214001400, magic: 0x0208020082830080, shift: 57, offset: 768 },
    MagicEntry { mask: 0x0000024428002800, magic: 0x0804200202010014, shift: 57, offset: 896 },
    MagicEntry { mask: 0x0002040850005000, magic: 0x1110200110101000, shift: 57, offset: 1024 },
    MagicEntry { mask: 0x0004081020002000, magic: 0x0011000401015000, shift: 59, offset: 1152 },
    MagicEntry { mask: 0x0008102040004000, magic: 0x2001000024010410, shift: 59, offset: 1184 },
    MagicEntry { mask: 0x0008040200020400, magic: 0x00084000880208C8, shift: 59, offset: 1216 },
    MagicEntry { mask: 0x0010080400040800, magic: 0x2402280802080820, shift: 59, offset: 1248 },
    MagicEntry { mask: 0x0020100A000A1000, magic: 0x02080C0006002602, shift: 57, offset: 1280 },
    MagicEntry { mask: 0x0040221400142200, magic: 0x0002008008008102, shift: 55, offset: 1408 },
    MagicEntry { mask: 0x0002442800284400, magic: 0x0803010000444000, shift: 55, offset: 1920 },
    MagicEntry { mask: 0x0004085000500800, magic: 0x0201020008405004, shift: 57, offset: 2432 },
    MagicEntry { mask: 0x0008102000201000, magic: 0x2844204084821088, shift: 59, offset: 2560 },
    MagicEntry { mask: 0x0010204000402000, magic: 0x00C0484021010800, shift: 59, offset: 2592 },
    MagicEntry { mask: 0x0004020002040800, magic: 0x020C10480014A024, shift: 59, offset: 2624 },
    MagicEntry { mask: 0x0008040004081000, magic: 0x0024040400031040, shift: 59, offset: 2656 },
    MagicEntry { mask: 0x00100A000A102000, magic: 0x2009040210010804, shift: 57, offset: 2688 },
    MagicEntry { mask: 0x0022140014224000, magic: 0x8000200800090106, shift: 55, offset: 2816 },
    MagicEntry { mask: 0x0044280028440200, magic: 0x1811010401420020, shift: 55, offset: 3328 },
    MagicEntry { mask: 0x0008500050080400, magic: 0x1E05080121020200, shift: 57, offset: 3840 },
    MagicEntry { mask: 0x0010200020100800, magic: 0x0028820880040894, shift: 59, offset: 3968 },
    MagicEntry { mask: 0x0020400040201000, magic: 0x010C108200023500, shift: 59, offset: 4000 },
    MagicEntry { mask: 0x0002000204081000, magic: 0x0050900808C22011, shift: 59, offset: 4032 },
    MagicEntry { mask: 0x0004000408102000, magic: 0x6520881110110901, shift: 59, offset: 4064 },
    MagicEntry { mask: 0x000A000A10204000, magic: 0x0007010801080200, shift: 57, offset: 4096 },
    MagicEntry { mask: 0x0014001422400000, magic: 0x20B0012018043100, shift: 57, offset: 4224 },
    MagicEntry { mask: 0x0028002844020000, magic: 0x1446082100404405, shift: 57, offset: 4352 },
    MagicEntry { mask: 0x0050005008040200, magic: 0x4C81020081001200, shift: 57, offset: 4480 },
    MagicEntry { mask: 0x0020002010080400, magic: 0x1888C80800800044, shift: 59, offset: 4608 },
    MagicEntry { mask: 0x0040004020100800, magic: 0x0510010101080020, shift: 59, offset: 4640 },
    MagicEntry { mask: 0x0000020408102000, magic: 0x500400A290100000, shift: 59, offset: 4672 },
    MagicEntry { mask: 0x0000040810204000, magic: 0xC001006110080008, shift: 59, offset: 4704 },
    MagicEntry { mask: 0x00000A1020400000, magic: 0x0000020100886140, shift: 59, offset: 4736 },
    MagicEntry { mask: 0x0000142240000000, magic: 0x0080002042020000, shift: 59, offset: 4768 },
    MagicEntry { mask: 0x0000284402000000, magic: 0xA000101002088044, shift: 59, offset: 4800 },
    MagicEntry { mask: 0x0000500804020000, magic: 0x8311100210010302, shift: 59, offset: 4832 },
    MagicEntry { mask: 0x0000201008040200, magic: 0xC804084204040040, shift: 59, offset: 4864 },
    MagicEntry { mask: 0x0000402010080400, magic: 0x0009010102160188, shift: 59, offset: 4896 },
    MagicEntry { mask: 0x0002040810204000, magic: 0x400621004A304004, shift: 58, offset: 4928 },
    MagicEntry { mask: 0x0004081020400000, magic: 0x8000044C02011000, shift: 59, offset: 4992 },
    MagicEntry { mask: 0x000A102040000000, magic: 0x1004090080480800, shift: 59, offset: 5024 },
    MagicEntry { mask: 0x0014224000000000, magic: 0x0001808008460810, shift: 59, offset: 5056 },
    MagicEntry { mask: 0x0028440200000000, magic: 0xA489500090320880, shift: 59, offset: 5088 },
    MagicEntry { mask: 0x0050080402000000, magic: 0x8002000450820200, shift: 59, offset: 5120 },
    MagicEntry { mask: 0x0020100804020000, magic: 0x0200850404280200, shift: 59, offset: 5152 },
    MagicEntry { mask: 0x0040201008040200, magic: 0x0602040414084202, shift: 58, offset: 5184 },
];

const ROOK_MAP_SIZE: usize = 102400;
const BISHOP_MAP_SIZE: usize = 5248;

// shift helpers (build/attacks.rs)
const A_FILE: u64 = 0x101010101010101;
const H_FILE: u64 = A_FILE << 7;

fn shift_dir(mut bb: u64, dir: i8) -> u64 {
    let file_offset = dir & 0x7;
    if file_offset == 1 {
        bb &= !H_FILE;
    } else if file_offset == 7 {
        bb &= !A_FILE;
    }
    if dir < 0 { bb >> -dir } else { bb << dir }
}
fn shift_dirs(bb: u64, dirs: &[i8]) -> u64 {
    let mut t = 0;
    for &d in dirs {
        t |= shift_dir(bb, d);
    }
    t
}
fn pawn_attacks_gen(sq: u8, white: bool) -> u64 {
    if white { shift_dirs(1 << sq, &[7, 9]) } else { shift_dirs(1 << sq, &[-7, -9]) }
}
fn king_attacks_gen(sq: u8) -> u64 {
    shift_dirs(1 << sq, &[7, 8, 9, 1, -7, -8, -9, -1])
}
fn knight_attacks_gen(sq: u8) -> u64 {
    let t = shift_dirs(1 << sq, &[7, 9, -7, -9]);
    let t = shift_dirs(t, &[8, 1, -8, -1]);
    t & !king_attacks_gen(sq)
}
fn generate_slide(sq: u8, occ: u64, dir: i8) -> u64 {
    let mut t = shift_dir(1 << sq, dir);
    for _ in 0..8 {
        if t & occ != 0 {
            break;
        }
        t |= shift_dir(t, dir);
    }
    t
}
fn sliding_attacks(sq: u8, occ: u64, dirs: &[i8]) -> u64 {
    dirs.iter().fold(0, |o, &d| o | generate_slide(sq, occ, d))
}

fn magic_index(occ: u64, e: &MagicEntry) -> usize {
    let mut hash = occ & e.mask;
    hash = hash.wrapping_mul(e.magic) >> e.shift;
    hash as usize + e.offset
}

struct Tables {
    king: [u64; 64],
    knight: [u64; 64],
    pawn: [[u64; 64]; 2],
    rook: Vec<u64>,
    bishop: Vec<u64>,
}

static TABLES: OnceLock<Tables> = OnceLock::new();
static THREAT_LUT: OnceLock<ThreatLut> = OnceLock::new();

fn tables() -> &'static Tables {
    TABLES.get_or_init(|| {
        let mut king = [0u64; 64];
        let mut knight = [0u64; 64];
        let mut pawn = [[0u64; 64]; 2];
        for s in 0..64 {
            king[s] = king_attacks_gen(s as u8);
            knight[s] = knight_attacks_gen(s as u8);
            pawn[0][s] = pawn_attacks_gen(s as u8, true);
            pawn[1][s] = pawn_attacks_gen(s as u8, false);
        }
        let rook = gen_sliding_map(ROOK_MAP_SIZE, &ROOK_MAGICS, &[8, -8, 1, -1]);
        let bishop = gen_sliding_map(BISHOP_MAP_SIZE, &BISHOP_MAGICS, &[9, 7, -7, -9]);
        Tables { king, knight, pawn, rook, bishop }
    })
}

fn gen_sliding_map(size: usize, magics: &[MagicEntry], dirs: &[i8]) -> Vec<u64> {
    let mut map = vec![0u64; size];
    for square in 0..64u8 {
        let e = &magics[square as usize];
        let mut occ = 0u64;
        let perms = 1u64 << e.mask.count_ones();
        for _ in 0..perms {
            let hash = magic_index(occ, e);
            map[hash] = sliding_attacks(square, occ, dirs);
            occ = occ.wrapping_sub(e.mask) & e.mask;
        }
    }
    map
}

fn pawn_attacks(sq: Square, c: Color) -> Bitboard {
    Bitboard(tables().pawn[c as usize][sq.0 as usize])
}
fn king_attacks(sq: Square) -> Bitboard {
    Bitboard(tables().king[sq.0 as usize])
}
fn knight_attacks(sq: Square) -> Bitboard {
    Bitboard(tables().knight[sq.0 as usize])
}
fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let e = &ROOK_MAGICS[sq.0 as usize];
    Bitboard(tables().rook[magic_index(occ.0, e)])
}
fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let e = &BISHOP_MAGICS[sq.0 as usize];
    Bitboard(tables().bishop[magic_index(occ.0, e)])
}
fn queen_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    Bitboard(rook_attacks(sq, occ).0 | bishop_attacks(sq, occ).0)
}
fn attacks(piece: Piece, sq: Square, occ: Bitboard) -> Bitboard {
    match piece.piece_type() {
        PieceType::Pawn => pawn_attacks(sq, piece.color()),
        PieceType::Knight => knight_attacks(sq),
        PieceType::Bishop => bishop_attacks(sq, occ),
        PieceType::Rook => rook_attacks(sq, occ),
        PieceType::Queen => queen_attacks(sq, occ),
        PieceType::King => king_attacks(sq),
    }
}

// ───────────────────── threat index (threats/threat_index.rs) ─────────────────────

#[derive(Copy, Clone)]
struct PiecePair {
    inner: u32,
}
impl PiecePair {
    const fn new(excluded: bool, semi_excluded: bool, base: i32) -> Self {
        Self {
            inner: (((semi_excluded && !excluded) as u32) << 30)
                | ((excluded as u32) << 31)
                | ((base & 0x3FFFFFFF) as u32),
        }
    }
    const fn base(self, attacking: Square, attacked: Square) -> isize {
        let below = (attacking.0 < attacked.0) as u32;
        ((self.inner.wrapping_add(below << 30)) & 0x80FFFFFF) as i32 as isize
    }
}

struct ThreatLut {
    piece_pair: [[PiecePair; 12]; 12],
    piece_offset: [[i32; 64]; 12],
    attack_index: [[[u8; 64]; 64]; 12],
}

fn threat_lut() -> &'static ThreatLut {
    THREAT_LUT.get_or_init(|| {
        #[rustfmt::skip]
        const PIECE_INTERACTION_MAP: [[i32; 6]; 6] = [
            [0,  1, -1,  2, -1, -1],
            [0,  1,  2,  3,  4, -1],
            [0,  1,  2,  3, -1, -1],
            [0,  1,  2,  3, -1, -1],
            [0,  1,  2,  3,  4, -1],
            [0,  1,  2,  3, -1, -1],
        ];
        const PIECE_TARGET_COUNT: [i32; 6] = [6, 10, 8, 8, 10, 8];

        let mut piece_pair = [[PiecePair { inner: 0 }; 12]; 12];
        let mut piece_offset_lut = [[0i32; 64]; 12];
        let mut attack_index = [[[0u8; 64]; 64]; 12];

        let mut offset = 0i32;
        let mut piece_offset = [0i32; Piece::NUM];
        let mut offset_table = [0i32; Piece::NUM];

        for piece_color in [Color::White, Color::Black] {
            for pt_i in 0..PieceType::NUM {
                let pt = PieceType::new(pt_i);
                let piece = Piece::new(piece_color, pt);

                let mut count = 0i32;
                for square in 0..64usize {
                    piece_offset_lut[piece.0 as usize][square] = count;
                    if pt != PieceType::Pawn || (8..56).contains(&square) {
                        count += attacks(piece, Square::new(square as u8), Bitboard(0)).popcount() as i32;
                    }
                }
                piece_offset[piece.0 as usize] = count;
                offset_table[piece.0 as usize] = offset;
                offset += PIECE_TARGET_COUNT[pt as usize] * count;
            }
        }

        for attacking in Piece::ALL {
            for attacked in Piece::ALL {
                let ap = attacking.piece_type();
                let ac = attacking.color();
                let dp = attacked.piece_type();
                let dc = attacked.color();

                let map = PIECE_INTERACTION_MAP[ap as usize][dp as usize];
                let base = offset_table[attacking.0 as usize]
                    + ((dc as i32) * (PIECE_TARGET_COUNT[ap as usize] / 2) + map)
                        * piece_offset[attacking.0 as usize];

                let enemy = ac != dc;
                let semi_excluded = ap == dp && (enemy || ap != PieceType::Pawn);
                let excluded = map < 0;

                piece_pair[attacking.0 as usize][attacked.0 as usize] =
                    PiecePair::new(excluded, semi_excluded, base);
            }
        }

        for piece in Piece::ALL {
            for from in 0..64usize {
                let atk = attacks(piece, Square::new(from as u8), Bitboard(0));
                for to in 0..64usize {
                    let mask = if to == 0 { 0u64 } else { (1u64 << to) - 1 };
                    attack_index[piece.0 as usize][from][to] = (Bitboard(mask) & atk).popcount() as u8;
                }
            }
        }

        ThreatLut { piece_pair, piece_offset: piece_offset_lut, attack_index }
    })
}

fn threat_index(piece: Piece, from: Square, attacked: Piece, to: Square, mirrored: bool, pov: Color) -> isize {
    let from = Square(from.relative_to(pov).0 ^ (7 * mirrored as u8));
    let to = Square(to.relative_to(pov).0 ^ (7 * mirrored as u8));

    let attacking = (piece.0 as usize) ^ (pov as usize);
    let attacked = (attacked.0 as usize) ^ (pov as usize);

    let lut = threat_lut();
    let pair = lut.piece_pair[attacking][attacked];
    pair.base(from, to)
        + lut.piece_offset[attacking][from.0 as usize] as isize
        + lut.attack_index[attacking][from.0 as usize][to.0 as usize] as isize
}

// ───────────────────── pst index (psq.rs / dump.rs) ─────────────────────

fn pst_index(color: Color, piece: PieceType, square: Square, king: Square, pov: Color) -> usize {
    let flip = (7 * (king.is_kingside() as u8)) ^ (56 * (pov as u8));
    let bucket = INPUT_BUCKETS_LAYOUT[((king.0) ^ flip) as usize] as usize;
    bucket * 768 + 384 * (color != pov) as usize + 64 * (piece as usize) + (((square.0) ^ flip) as usize)
}

// ───────────────────── board reconstruction ─────────────────────

/// Absolute (non-STM-relative) board: per-square piece array + occupancy.
#[derive(Clone)]
pub struct AresBoard {
    occ: u64,
    piece_on: [Piece; 64], // valid only where occ bit set
    king: [Square; 2],     // [white, black]
}

impl AresBoard {
    /// Build from absolute bitboards in engine order:
    /// pieces12[Piece index 0..12] where index = (piece_type<<1)|color (matches `Piece`).
    pub fn from_absolute(pieces12: [u64; 12]) -> Self {
        let mut occ = 0u64;
        let mut piece_on = [Piece(0); 64];
        let mut king = [Square(0); 2];
        for pi in 0..12usize {
            let mut bb = pieces12[pi];
            occ |= bb;
            let piece = Piece::from_index(pi);
            while bb != 0 {
                let sq = bb.trailing_zeros() as u8;
                bb &= bb - 1;
                piece_on[sq as usize] = piece;
                if piece.piece_type() == PieceType::King {
                    king[piece.color() as usize] = Square(sq);
                }
            }
        }
        AresBoard { occ, piece_on, king }
    }

    /// Reconstruct the ABSOLUTE board from a side-to-move-relative
    /// `bulletformat::ChessBoard`. `stm_is_black` un-flips the relative encoding.
    pub fn from_bullet(board: &ChessBoard, stm_is_black: bool) -> Self {
        let mut pieces12 = [0u64; 12];
        for (piece, square) in (*board).into_iter() {
            // bulletformat: bit 3 = color (0=stm,1=ntm relative), bits0..2 = piece type
            let rel_color = (piece >> 3) & 1; // 0 = stm-relative-white
            let pt = (piece & 7) as usize; // 0..5
            let mut sq = square;
            let mut color = rel_color;
            if stm_is_black {
                sq ^= 56;
                color ^= 1;
            }
            let idx = (pt << 1) | color as usize;
            pieces12[idx] |= 1u64 << sq;
        }
        Self::from_absolute(pieces12)
    }

    fn king_square(&self, pov: Color) -> Square {
        self.king[pov as usize]
    }
}

/// Active feature indices (combined PST+threat space) for one perspective —
/// replicates `tools/dump.rs::features_for_pov` exactly. Calls `emit` per feature.
fn features_for_pov<F: FnMut(usize)>(board: &AresBoard, pov: Color, mut emit: F) {
    let king = board.king_square(pov);
    let mirrored = king.is_kingside();
    let occ = Bitboard(board.occ);

    let mut sq_iter = Bitboard(board.occ);
    while let Some(square) = sq_iter.next() {
        let piece = board.piece_on[square.0 as usize];

        emit(pst_index(piece.color(), piece.piece_type(), square, king, pov));

        let threats = attacks(piece, square, occ) & occ;
        for target in threats {
            let attacked = board.piece_on[target.0 as usize];
            let index = threat_index(piece, square, attacked, target, mirrored, pov);
            if index >= 0 {
                emit(THREAT_BASE + index as usize);
            }
        }
    }
}

// ───────────────────── SparseInputType ─────────────────────

/// Ares combined PST+threat features, bit-for-bit identical to the engine.
///
/// `map_features` emits `f(white_index, black_index)` so that POV0 = White
/// perspective and POV1 = Black perspective (matching the corpus oracle).
///
/// NOTE: `RequiredDataType` is `AresChessBoard`, a thin wrapper carrying the
/// absolute board plus a flag recording whether the original STM was Black,
/// because a bare `bulletformat::ChessBoard` is STM-relative and cannot by
/// itself recover the absolute frame the corpus uses.
#[derive(Clone, Copy, Debug, Default)]
pub struct AresThreats;

/// Data type fed to the trainer: an absolute board (reconstructed from the
/// STM-relative bulletformat board + known STM).
#[derive(Clone, Copy)]
pub struct AresChessBoard {
    pub board: ChessBoard,
    pub stm_is_black: bool,
}

unsafe impl Send for AresChessBoard {}
unsafe impl Sync for AresChessBoard {}

impl SparseInputType for AresThreats {
    type RequiredDataType = AresChessBoard;

    fn num_inputs(&self) -> usize {
        NUM_INPUTS
    }

    fn max_active(&self) -> usize {
        // Measured over the 644-position parity corpus: max 84 features/POV (32 PST + threats),
        // mean ~47. 256 is a safe upper bound with wide margin for dense positions, and 8x smaller
        // than the old 2048 — that bound sizes a per-batch Vec<i32> (max_active*batch_size) that is
        // memset every batch, so an oversized value directly throttles throughput.
        256
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        // Correct dual-perspective emission.
        //
        // The bullet sparse encoding writes `our` into the STM accumulator and `opp`
        // into the NTM accumulator at the SAME slot (see value.rs): the two columns are
        // INDEPENDENT active-feature lists, coupled only by their slot count. The only
        // requirement is that the STM list be EXACTLY the engine's mover-POV active set
        // and the NTM list be EXACTLY the engine's non-mover-POV active set; the pairing
        // order is irrelevant because the sparse accumulator sums columns set-wise.
        //
        // The previous implementation tried to pair each active feature across both POVs
        // at once and, for threats that the engine includes in only ONE perspective
        // (a threat index can be < 0 / excluded for one POV but valid for the other),
        // fell back to a SELF-PAIR `f(idx, idx)` — which injected that feature into BOTH
        // accumulators, polluting the perspective the engine deliberately excluded it
        // from. That produced spurious threat features (verified against the corpus) for
        // BOTH white- and black-to-move positions and is what broke the trained net.
        //
        // Fix: enumerate each perspective independently via the SAME `features_for_pov`
        // path the parity gate / engine oracle use, then zip the two sorted sets. The
        // engine guarantees equal per-POV active counts (PST: one per piece; threats:
        // exclusions are symmetric in count), so the zip is total — verified across the
        // full 644-position corpus.
        //
        // POV0 == side-to-move. For a STM-relative bulletformat board (`stm_is_black:
        // false`, as the trainer feeds), the reconstructed board has the mover as White,
        // so STM = White-POV and NTM = Black-POV on that reconstructed board.
        let board = AresBoard::from_bullet(&pos.board, pos.stm_is_black);
        let (stm, ntm) =
            if pos.stm_is_black { (Color::Black, Color::White) } else { (Color::White, Color::Black) };

        let mut stm_feats: Vec<usize> = Vec::with_capacity(64);
        let mut ntm_feats: Vec<usize> = Vec::with_capacity(64);
        features_for_pov(&board, stm, |i| stm_feats.push(i));
        features_for_pov(&board, ntm, |i| ntm_feats.push(i));
        stm_feats.sort_unstable();
        ntm_feats.sort_unstable();

        debug_assert_eq!(
            stm_feats.len(),
            ntm_feats.len(),
            "Ares per-POV active feature counts must match for the dual-perspective zip"
        );

        for (&s, &n) in stm_feats.iter().zip(ntm_feats.iter()) {
            f(s, n);
        }
    }

    fn shorthand(&self) -> String {
        format!("{NUM_INPUTS}")
    }

    fn description(&self) -> String {
        "Ares combined PST (king-bucketed) + threat features, engine-parity".to_string()
    }
}

// ───────────────────── trainer plumbing for AresChessBoard ─────────────────────

/// AresChessBoard is the trainer data type. It wraps a `bulletformat::ChessBoard`,
/// so score/result delegate straight through to the inner (already STM-relative)
/// board, matching the ChessBoard blanket impl.
impl LoadableDataType for AresChessBoard {
    fn score(&self) -> i16 {
        BulletFormat::score(&self.board)
    }

    fn result(&self) -> GameResult {
        [GameResult::Loss, GameResult::Draw, GameResult::Win][self.board.result_idx()]
    }
}

/// Engine output-bucket table (Ares `OUTPUT_BUCKETS_LAYOUT`, indexed by board
/// occupancy popcount 0..=32). This is NOT `MaterialCount` — that divides the
/// popcount uniformly and diverges from the engine table in 13/31 buckets.
#[rustfmt::skip]
pub const OUTPUT_BUCKETS_LAYOUT: [u8; 33] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3,
    4, 4, 4,
    5, 5, 5,
    6, 6, 6,
    7, 7, 7, 7,
];

/// `OutputBuckets` for the Ares trainer: returns the engine bucket for a board
/// directly from its occupancy popcount via `OUTPUT_BUCKETS_LAYOUT`.
#[derive(Clone, Copy, Default)]
pub struct AresOutputBuckets;

impl OutputBuckets<AresChessBoard> for AresOutputBuckets {
    const BUCKETS: usize = 8;

    fn bucket(&self, pos: &AresChessBoard) -> u8 {
        OUTPUT_BUCKETS_LAYOUT[pos.board.occ().count_ones() as usize]
    }
}

impl AresThreats {
    /// Collect the (stm_idx, ntm_idx) feature pairs for a training position, in
    /// the SAME order `map_features` emits them. This is the precompute entry
    /// point: it reuses the exact `SparseInputType::map_features` logic (no
    /// reimplementation) so the premapped records are bit-identical to the
    /// on-the-fly path.
    pub fn map_pairs(board: &AresChessBoard) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        AresThreats.map_features(board, |s, n| out.push((s, n)));
        out
    }

    /// Collect the active feature indices for a given absolute board + perspective.
    /// This is the parity-gate entry point (POV0=White, POV1=Black).
    pub fn features(board: &AresChessBoard, pov_white: bool) -> Vec<usize> {
        let abs = AresBoard::from_bullet(&board.board, board.stm_is_black);
        let pov = if pov_white { Color::White } else { Color::Black };
        let mut out = Vec::new();
        features_for_pov(&abs, pov, |i| out.push(i));
        out.sort_unstable();
        out
    }
}
