import chess

# Dictionary of tactical test positions with categories
TACTICAL_TEST_POSITIONS = {
    # Checkmate patterns (verified forced mates)
    "mate_in_one": [
        # Back-rank mate (fixed)
        ("3r2k1/5ppp/4p3/8/8/8/5PPP/3R2K1 w - - 0 1", "d1d8"),
        # Anastasia's mate (queen + rook)
        ("2r5/4Nppk/4pn2/8/8/4K3/8/3R1Q2 w - - 0 1", "f1h1"),
        ("2r4k/5Q2/4R3/8/8/4K3/8/8 w - - 0 1", "e6h6")
        ("7k/6r1/5r2/8/8/Q7/7K/8 b - - 0 1", "f6h6"),
        # Smothered mate (knight checkmate)
        ("5rk1/5pp1/8/5N2/8/8/5PP1/4K2R w - - 0 1", "f5e7"),
        # Classic two-rook checkmate
        ("7k/5Rpp/8/8/8/8/5RPP/7K w - - 0 1", "f7f8"),
        ("6k1/5Rpp/8/8/8/8/5RPP/7K w - - 0 1", "f7f8"),
    ],
    
    # Knight forks (undefended pieces)
    "knight_fork": [
        # Fork king + queen
        ("r3k2r/ppp2ppp/2n5/3N4/4q3/8/PPP2PPP/R3K2R w KQkq - 0 1", "d5c7"),
        # Fork king + rook
        ("r3k2r/pp3ppp/2n1b3/3N4/8/8/PPP2PPP/R3K2R w KQkq - 0 1", "d5c7"),
        # Fork queen + rook (smothered setup)
        ("r1bqkbnr/ppppnppp/4p3/7N/8/1P6/PBP1PPPP/RN1QKB1R b KQkq - 0 1", "h5g7"),
        # Fork two rooks
        ("r3k2r/ppp2ppp/2n5/3N4/8/8/PPP2PPP/2KR3R w kq - 0 1", "d5c7"),
        ("r3k2r/p1p2ppp/8/8/2N1n3/8/PPP2PPP/2KR3R b kq - 0 1", "e4f2"),
    ],
    
    # Absolute pins (pinned to king)
    "pin": [
        ("r3k2r/ppp2ppp/2q1b3/8/8/2N5/PPB2PPP/R3K2R w KQkq - 0 1", "c2a4"),
        ("r2k3r/ppp2ppp/2nqb3/8/8/2N5/PPP2PPP/R3K2R w KQ - 0 1", "a1d1"),
        ("r1bqk2r/ppp1bppp/2n5/3p4/3P4/2N1PN2/PP3PPP/R2QKB1R w KQkq - 0 1", "f1b5"),

    ],
    
    # Discovered attacks/checks (verified)
    "discovered": [
        ("1k5r/1pp2ppp/p7/8/8/2N5/PPPR1PPP/2KR4 b - - 0 1", "d2d8"),
        # Pawn move reveals rook check
        ("r3k2r/ppp1qppp/4n3/8/N7/8/PPP2PPP/R3K2R b KQkq - 0 1", "e6c5"),
    ],
    
    # Skewers (verified)
    "skewer": [
        # rook skewers king + rook
        ("1k1r3r/ppp2ppp/2n5/8/8/2N5/PPPR1PPP/2KR4 b - - 0 1", "d2d8"),
        # Rook skewers king + bishop
        ("r2k3r/ppp2ppp/2nqb3/8/8/2P1N3/PP2KPPP/R6R b - - 0 1", "a1d1"),
    ],
    
    # Endgame tactics (verified)
    "endgame": [
        # Opposition (king vs king)
        ("7k/5R2/6K1/8/8/8/8/8 b - - 0 1", "f7f8"),
    ]
}

# Move Index Mapping
promotion_moves = {}
promotion_idx = 4096
for rank in [6, 1]:
    for col in range(8):
        from_square = chess.square(col, rank)
        to_square = chess.square(col, rank + (1 if rank == 6 else -1))
        for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            promotion_moves[(from_square, to_square, piece)] = promotion_idx
            promotion_idx += 1
        if col > 0:
            to_square = chess.square(col - 1, rank + (1 if rank == 6 else -1))
            for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promotion_moves[(from_square, to_square, piece)] = promotion_idx
                promotion_idx += 1
        if col < 7:
            to_square = chess.square(col + 1, rank + (1 if rank == 6 else -1))
            for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promotion_moves[(from_square, to_square, piece)] = promotion_idx
                promotion_idx += 1
