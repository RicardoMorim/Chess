import chess

# Dictionary of tactical test positions with categories
TACTICAL_TEST_POSITIONS = {
    # Checkmate patterns
    "mate_in_one": [
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1", "h5f7"),
        ("r1bq2r1/ppp1bpkp/2np1np1/4p3/2B1P3/2NP1N2/PPPBQPPP/R3K2R w KQ - 0 1", "d2h6"),
        ("r3k2r/ppp2p1p/2n1bN2/2b1P1p1/2p1q3/2P5/PP1Q1PPP/RNB1K2R w KQkq - 0 1", "d2d8"),
    ],
    
    # Knight forks
    "knight_fork": [
        ("r3k2r/ppp2ppp/2n5/3Nn3/8/8/PPP2PPP/R3K2R w KQkq - 0 1", "d5f6"),
        ("rnbqk2r/ppp1bppp/3p1n2/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w kq - 0 1", "f3e5"),
        ("r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", "f3e5"),
    ],
    
    # Pin patterns
    "pin": [
        ("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", "c4f7"),
        ("rnbqkb1r/pp3ppp/2p1pn2/3p4/3P4/2NBPN2/PPP2PPP/R1BQK2R w KQkq - 0 1", "c3e5"),
        ("r1bqk2r/ppp2ppp/2n2n2/1B1pp3/1b2P3/2N2N2/PPPP1PPP/R1BQ1RK1 w kq - 0 1", "b5e8"),
    ],
    
    # Discovered attacks/checks
    "discovered": [
        ("rnbqkbnr/pppp1ppp/8/4p3/3P4/2N5/PPP1PPPP/R1BQKBNR b KQkq - 0 1", "e5d4"),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", "c4d5"),
        ("r1bqk2r/ppp1bppp/2n2n2/3pp3/2BP4/2N1PN2/PP3PPP/R1BQK2R w KQkq - 0 1", "d4e5"),
    ],
    
    # Skewers
    "skewer": [
        ("r1bqk1nr/ppp2ppp/2n5/3p4/1bBP4/2N5/PPP2PPP/R1BQK1NR w KQkq - 0 1", "c1g5"),
        ("r3k2r/pp3ppp/2p1bn2/q2p4/3P4/2PBP3/PP1N1PPP/R2QK2R b KQkq - 0 1", "e6a2"),
        ("r1b1kb1r/pp3ppp/2nqpn2/3p4/3P4/2N1PN2/PP3PPP/R1BQK2R w KQkq - 0 1", "f1b5"),
    ],
    
    # Endgame tactics
    "endgame": [
        ("8/8/1KP5/3r4/8/8/8/k7 w - - 0 1", "c6c7"),
        ("8/4kp2/2p3p1/1p2P1P1/8/2P3K1/8/8 w - - 0 1", "g3f4"),
        ("8/5p2/5k2/p1p2F2/Pp6/1P4K1/8/8 w - - 0 1", "g3f4"),
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
