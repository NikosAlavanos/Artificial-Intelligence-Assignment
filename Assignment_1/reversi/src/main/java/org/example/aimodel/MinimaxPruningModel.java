package org.example.aimodel;

import org.example.board.Color;
import org.example.board.GameState;
import org.example.game.Move;
import org.example.game.Player;
import org.example.heuristic.CornerHeuristicFunction;
import org.example.heuristic.HeuristicFunction;

import java.util.ArrayList;
import java.util.Random;

public class MinimaxPruningModel extends AiModel {
    private final int maxDepth;
    private Random r = new Random();
    private HeuristicFunction heuristic = new CornerHeuristicFunction();

    public MinimaxPruningModel(int maxDepth) {
        this.maxDepth = maxDepth;
    }

    @Override
    public Move thinkMove(Player player, GameState state) {
        if (player.getColor() == Color.BLACK) { // Max
            return max(state, 0, Integer.MIN_VALUE, Integer.MAX_VALUE);
        } else {
            return min(state, 0, Integer.MIN_VALUE, Integer.MAX_VALUE);
        }
    }

    private Move max(GameState state, int depth, int alpha, int beta) {
        if (state.isGameOver() || (depth == this.maxDepth)) {
            return new Move(state.getLastMove().getRow(), state.getLastMove().getCol(), heuristic.evaluate(state));
        }

        ArrayList<GameState> children = state.getChildren(Color.BLACK);
        Move maxMove = new Move(Integer.MIN_VALUE); // put max node initially to smallest value.

        for(GameState child: children) {
            Move move = min(child, depth + 1, alpha, beta);

            if (move.getScore() >= maxMove.getScore()) {
                if((move.getScore()) == maxMove.getScore()) {
                    if(r.nextInt(2) == 0) {
                        maxMove.setRow(child.getLastMove().getRow());
                        maxMove.setCol(child.getLastMove().getCol());
                        maxMove.setScore(move.getScore());
                    }
                } else {
                    maxMove.setRow(child.getLastMove().getRow());
                    maxMove.setCol(child.getLastMove().getCol());
                    maxMove.setScore(move.getScore());
                }
            }

            if (maxMove.getScore() < alpha) {
                return maxMove;
            }
            beta = Math.min(beta, maxMove.getScore());
        }
        return maxMove;
    }

    private Move min(GameState state, int depth, int alpha, int beta) {
        if(state.isGameOver() || (depth == this.maxDepth))         {
            return new Move(state.getLastMove().getRow(), state.getLastMove().getCol(), heuristic.evaluate(state));
        }

        ArrayList<GameState> children = state.getChildren(Color.WHITE);
        Move minMove = new Move(Integer.MAX_VALUE);

        for(GameState child: children) {
            Move move = max(child, depth + 1, alpha, beta);

            if(move.getScore() <= minMove.getScore()) {
                if((move.getScore()) == minMove.getScore())
                {
                    if(r.nextInt(2) == 0)
                    {
                        minMove.setRow(child.getLastMove().getRow());
                        minMove.setCol(child.getLastMove().getCol());
                        minMove.setScore(move.getScore());
                    }
                }
                else
                {
                    minMove.setRow(child.getLastMove().getRow());
                    minMove.setCol(child.getLastMove().getCol());
                    minMove.setScore(move.getScore());
                }
            }

            if (minMove.getScore() > beta) {
                return minMove;
            }
            alpha = Math.max(alpha, minMove.getScore());
        }
        return minMove;
    }
}
