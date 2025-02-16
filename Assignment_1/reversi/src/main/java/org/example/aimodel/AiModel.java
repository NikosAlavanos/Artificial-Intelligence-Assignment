package org.example.aimodel;

import org.example.board.GameState;
import org.example.game.Move;
import org.example.game.Player;

public abstract class AiModel {
    public abstract Move thinkMove(Player player, GameState state);
}
