package org.example.aimodel;

import org.example.board.GameState;
import org.example.game.Move;
import org.example.game.Player;

import java.util.List;

public class HillClimbingModel extends AiModel {
    private static final int[][] boardValues = {
            {10, 2, 7, 6, 6, 7, 2, 10},
            {2, 1, 3, 3, 3, 3, 1, 2},
            {7, 3, 6, 4, 4, 6, 3, 7},
            {6, 3, 4, 2, 2, 4, 3, 6},
            {6, 3, 4, 2, 2, 4, 3, 6},
            {7, 3, 6, 4, 4, 6, 3, 7},
            {2, 1, 3, 3, 3, 3, 1, 2},
            {10, 2, 7, 6, 6, 7, 2, 10}
    };

    private int evaluateMove(Move move) {
        return boardValues[move.getRow()][move.getCol()];
    }

    private Move getBestHillClimbingMove(List<Move> moves) {
        Move bestMove = null;
        int bestValue = Integer.MIN_VALUE;

        System.out.println("---------------------------------------");
        for (Move move : moves) {
            int value = evaluateMove(move);
            if (value > bestValue) {
                bestValue = value;
                bestMove = move;
            }

            System.out.println("move: " + move + ", score:" + value);
        }
        return bestMove;
    }


    @Override
    public Move thinkMove(Player player, GameState state) {
        List<Move> moves = state.getValidMoves(player.getColor());

        return getBestHillClimbingMove(moves);
    }

}
