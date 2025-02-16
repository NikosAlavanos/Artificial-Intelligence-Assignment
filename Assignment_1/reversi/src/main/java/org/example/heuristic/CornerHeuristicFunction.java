package org.example.heuristic;

import org.example.board.Color;
import org.example.board.GameState;

public class CornerHeuristicFunction extends HeuristicFunction {
    @Override
    public int evaluate(GameState state) {
        // state.discs[1][1].getColor()
        // state.activePlayer == Color.WHITE
        int score = 0;

        for (int i=0;i<8;i++) {
            for (int j=0;j<8;j++) {
                boolean isCorner;
                if ((i == 0 && j == 0) || (i == 7 && j == 7) || (i == 7 && j == 0) || (i == 0 && j == 7)) {
                    isCorner = true;
                } else {
                    isCorner = false;
                }
                if (state.discs[i][j] != null) {
                    if (state.discs[i][j].getColor() == Color.BLACK) {
                        if (isCorner) {
                            score += 30;
                        } else {
                            score ++;
                        }
                    } else {
                        if (isCorner) {
                            score -= 30;
                        } else {
                            score --;
                        }
                    }
                }
            }
        }

        return score;
    }
}
