package org.example.heuristic;

import org.example.board.Color;
import org.example.board.GameState;

public class SimpleHeuristicFunction extends HeuristicFunction {
    public int evaluate(GameState state) {
        int score = 0;

        for (int i=0;i<8;i++) {
            for (int j=0;j<8;j++) {
                if (state.discs[i][j] != null) {
                    if (state.discs[i][j].getColor() == Color.BLACK) {
                        score++;
                    } else {
                        score--;
                    }
                }
            }
        }

        return score;
    }
}
