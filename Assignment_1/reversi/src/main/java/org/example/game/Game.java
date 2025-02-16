package org.example.game;

import org.example.board.Color;
import org.example.board.Disc;
import org.example.board.GameState;

import static org.example.config.Configuration.DEBUG_MODE;

public class Game {
    private GameState state = null;
    public int CounterBlack = 0;
    public int CounterWhite = 0;

    public GameState getState() {
        return state;
    }

    public void start(Player black, Player white) {
        state = new GameState(black, white);

        if (DEBUG_MODE){
            state.discs[0][0] = new Disc(Color.BLACK);
            state.discs[1][1] = new Disc(Color.WHITE);
            state.discs[1][6] = new Disc(Color.WHITE);
            state.discs[0][7] = new Disc(Color.BLACK);
        } else {
            state.discs[3][3] = new Disc(Color.BLACK);
            state.discs[3][4] = new Disc(Color.WHITE);
            state.discs[4][3] = new Disc(Color.WHITE);
            state.discs[4][4] = new Disc(Color.BLACK);
        }


        state.activePlayer = Color.BLACK;

        System.out.println("Starting game with players: ");

        System.out.println(black);
        System.out.println(white);
    }

    public boolean isOver() {
        return state.isGameOver();

    }

    public void playMove(Move move) {
        state = state.playMove(move);
    }

    public void showTheWinner(){
        int[] scores = state.scoreCounter(CounterBlack, CounterWhite);
        CounterWhite = scores[0];
        CounterBlack = scores[1];
        if (CounterWhite > CounterBlack) {
            System.out.println("White is the winner with score: " + CounterWhite + ">" + CounterBlack);
        } else if (CounterBlack > CounterWhite) {
            System.out.println("Black is the winner with score: " + CounterBlack + ">" + CounterWhite);
        } else {
            System.out.println("It's a tie!" + CounterBlack + "=" + CounterWhite);
        }

        CounterWhite = 0;
        CounterBlack = 0;
    }


    public void setState(GameState newState) {
        this.state = newState;
    }

    public void print() {
        this.state.print();
    }

    public Move selectMove() {
        Color color = state.activePlayer;
        Player activePlayer;
        Player opponent;

        if (color == Color.BLACK) {
            activePlayer = state.black;
            opponent = state.white;
        } else {
            activePlayer = state.white;
            opponent = state.black;
        }

        if (activePlayer.isAi()) {
            Move move = activePlayer.thinkMove(state);
            return move;
        } else {
            Move move = state.getHumanMoveFromKeyboard(activePlayer.getColor());
            return move;
        }
    }
}
