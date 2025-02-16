package org.example.board;

import org.example.game.AiLevel;
import org.example.game.Move;
import org.example.game.Player;

import java.util.*;

public class GameState {
    public Disc[][] discs = new Disc[8][8];
    public Color activePlayer;
    public final Player black;
    public final Player white;
    private final Random random = new Random();
    private final Scanner scanner = new Scanner(System.in);
    GameState currentState = this;
    private Move lastmove = null;

    public GameState(Player black, Player white) {
        this.black = black;
        this.white = white;
        this.lastmove = new Move();

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                discs[i][j] = null;
            }
        }
    }

    public void print() {
        if (!isGameOver() && !getValidMoves(activePlayer).isEmpty()) {
            switch (activePlayer) {
                case WHITE:
                    System.out.println("White's turn");
                    break;
                case BLACK:
                    System.out.println("Black's turn");
                    break;
            }
            System.out.println("Valid Moves " + activePlayer + ":" + getValidMoves(activePlayer));

        }
//        System.out.println("Valid Moves White: " + getValidMoves(Color.WHITE) + (activePlayer == Color.WHITE));
        List<Move> validMoves = getValidMoves(activePlayer);

        Set<Move> validMovesSet = new HashSet<>(validMoves);

        System.out.print("  ");
        for (int i = 0; i < 8; i++) {
            System.out.print(" " + i);
        }
        System.out.println();
        for (int i = 0; i < 8; i++) {
            System.out.print(i + "  ");

            for (int j = 0; j < 8; j++) {
                Move currentMove = new Move(i, j);

                if (discs[i][j] == null) {
                    if (validMovesSet.contains(currentMove)) {
                        System.out.print("* ");
                    } else {
                        System.out.print(". ");
                    }
                } else if (discs[i][j].getColor() == Color.WHITE) {
                    System.out.print("W ");
                } else if (discs[i][j].getColor() == Color.BLACK) {
                    System.out.print("B ");
                } else {
                    System.out.print("? ");
                }
            }
            System.out.println();
        }
    }

    public GameState playMove(Move move) {
        GameState child = new GameState(black, white);


        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                child.discs[i][j] = currentState.discs[i][j];
            }
        }

        child.activePlayer = (activePlayer == Color.WHITE) ? Color.BLACK : Color.WHITE;

        if (move == null || move.getCol() < 0 || move.getCol() < 0) {
            return child;
        } else {
            child.discs[move.getRow()][move.getCol()] = new Disc(activePlayer);

            changeIntermediateDiscs(child, move, activePlayer);

            return child;
        }
    }

    public Move getHumanMoveFromKeyboard(Color playerColor) {
        List<Move> validMoves = getValidMoves(playerColor);

        if (!validMoves.isEmpty()) {
            while(true) {
                System.out.print("Choose column (0-7):");
                int col = scanner.nextInt();
                System.out.print("Choose row (0-7):");
                int row = scanner.nextInt();

                Move move = new Move(row, col);

                if (isValidMove(validMoves, move)) {
                    return move;
                } else {
                    System.out.println("This is not a valid move. Please select one of the valid moves");

                    System.out.println("Valid moves are: ");

                    for (Move m : validMoves) {
                        System.out.println(m);
                    }
                }
            }
        } else {
            System.out.println("You have no valid move. Press enter to skip your turn");
            scanner.nextLine();
            return null;
        }
    }

    public List<Move> getValidMoves(Color playerColor) {
        List<Move> validMoves = new ArrayList<>();

        int[][] directions = {
                {-1, -1}, {-1, 0}, {-1, 1},
                {0, -1}, {0, 1},
                {1, -1}, {1, 0}, {1, 1}
        };

        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                if (discs[row][col] == null) {
                    boolean isValid = false;

                    for (int[] direction : directions) {
                        int dRow = direction[0];
                        int dCol = direction[1];

                        if (capturesInDirection(row, col, dRow, dCol, playerColor)) {
                            isValid = true;
                            break;
                        }
                    }

                    if (isValid) {
                        validMoves.add(new Move(row, col));
                    }
                }
            }
        }
        return validMoves;

    }

    private boolean capturesInDirection(int row, int col, int dRow, int dCol, Color playerColor) {
        Color opponentColor = (playerColor == Color.WHITE) ? Color.BLACK : Color.WHITE;
        int currentRow = row + dRow;
        int currentCol = col + dCol;
        boolean hasOpponentDisc = false;

        while (currentRow >= 0 && currentRow < 8 && currentCol >= 0 && currentCol < 8) {
            Disc currentDisc = discs[currentRow][currentCol];

            if (currentDisc == null) {
                return false;
            }

            if (currentDisc.getColor() == opponentColor) {
                hasOpponentDisc = true;
            } else if (currentDisc.getColor() == playerColor) {
                return hasOpponentDisc;
            } else {
                return false;
            }

            currentRow += dRow;
            currentCol += dCol;
        }

        return false;
    }

    public boolean isValidMove(List<Move> validMoves, Move move) {
        for (Move valid : validMoves) {
            if (move.equals(valid)) {
                return true;
            }
        }
        return false;
    }

    private void changeIntermediateDiscs(GameState child, Move userMove, Color activePlayer) {
        int row = userMove.getRow();
        int col = userMove.getCol();

        int[][] directions = {
                {-1, -1}, {-1, 0}, {-1, 1},
                {0, -1}, {0, 1},
                {1, -1}, {1, 0}, {1, 1}
        };

        for (int[] direction : directions) {
            int dRow = direction[0];
            int dCol = direction[1];

            List<Move> discsToFlip = new ArrayList<>();
            int currentRow = row + dRow;
            int currentCol = col + dCol;

            while (currentRow >= 0 && currentRow < 8 && currentCol >= 0 && currentCol < 8) {
                Disc currentDisc = child.discs[currentRow][currentCol];

                if (currentDisc == null) {
                    break;
                }

                if (currentDisc.getColor() == activePlayer) {
                    for (Move move : discsToFlip) {
                        child.discs[move.getRow()][move.getCol()] = new Disc(activePlayer);
                    }
                    break;
                } else {
                    discsToFlip.add(new Move(currentRow, currentCol));
                }

                currentRow += dRow;
                currentCol += dCol;
            }
        }
    }

    public boolean isGameOver() {
        return getValidMoves(Color.BLACK).isEmpty() && getValidMoves(Color.WHITE).isEmpty();
    }

    public int[] scoreCounter(int CounterBlack, int CounterWhite) {

        for (Disc[] disc : discs) {
            for (Disc value : disc) {
                if (value != null) {
                    if (value.getColor() == Color.WHITE) {
                        CounterWhite++;
                    } else if (value.getColor() == Color.BLACK) {
                        CounterBlack++;
                    }
                }
            }
        }
        return new int[]{CounterWhite, CounterBlack};
    }

    public ArrayList<GameState> getChildren(Color color) {
        ArrayList<GameState> children = new ArrayList<>();

        List<Move> validMoves = getValidMoves(color);

        for (Move move : validMoves) {
            GameState childstate = playMove(move);
            childstate.lastmove = move;
            children.add(childstate);
        }
        return children;
    }

    public Move getLastMove() {
        return lastmove;
    }
}
