package org.example.heuristic;

import org.example.board.GameState;

public abstract class HeuristicFunction {
    public abstract int evaluate(GameState state);
}
