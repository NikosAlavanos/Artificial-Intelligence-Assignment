package org.example.game;

import org.example.aimodel.*;
import org.example.board.Color;
import org.example.board.GameState;
import org.example.config.Configuration;

public class Player {
    private int number;
    private Color color;
    private AiLevel ailevel;
    private AiModel iq;

    public Player(int number, Color color, boolean ai, AiLevel ailevel,int maxdepth) {
        this.number = number;
        this.color = color;
        this.ailevel = ai ? ailevel : null;

        if (ai && ailevel != null) {
            switch (ailevel) {
                case RANDOM:
                    iq = new RandomModel();
                    break;
                case HILL_CLIMBING:
                    iq = new HillClimbingModel();
                    break;
                case MINIMAX:
                    iq = new MinimaxModel(maxdepth);
                    break;
                case MINIMAX_ALPHA_BETA:
                    iq = new MinimaxPruningModel(maxdepth);
                    break;
                default:
                    iq = null;
            }
        } else {
            iq = null;
        }
    }

    public Color getColor() {
        return color;
    }

    public void setColor(Color color) {
        this.color = color;
    }

    public AiLevel getAilevel() {
        return ailevel;
    }

    public boolean isAi() {
        return iq != null;
    }

    public int getNumber() {
        return number;
    }

    public void setNumber(int number) {
        this.number = number;
    }

    @Override
    public String toString() {
        return "Player " + number + " " + color + " " + ((iq != null)?"AI":"HUMAN");
    }

    public Move thinkMove(GameState state) {
        return iq.thinkMove(this, state);
    }
}
