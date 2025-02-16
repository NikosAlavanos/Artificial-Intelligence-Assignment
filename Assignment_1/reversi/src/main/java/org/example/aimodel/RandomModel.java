package org.example.aimodel;

import org.example.board.GameState;
import org.example.game.Move;
import org.example.game.Player;

import java.util.List;
import java.util.Random;

public class RandomModel extends AiModel {
    private final Random random = new Random();

    @Override
    public Move thinkMove(Player player, GameState state) {
        List<Move> moves = state.getValidMoves(player.getColor());
        if(moves.size()>0) {
            int x = random.nextInt(moves.size());
            return moves.get(x);
        }else {
            return null;
        }
    }
}
