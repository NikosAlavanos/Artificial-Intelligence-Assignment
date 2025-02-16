package org.example.game;

import java.util.Objects;

public class Move {
    private int row, col;
    public int score;

    public Move(int score) {
        this.score = score;
    }

    public Move(int row, int col) {
        this.row = row;
        this.col = col;
    }

    public Move() {
        this.row = -1;
        this.col = -1;
        this.score = 0;
    }

    public Move(int row, int col, int score) {
        this.row = row;
        this.col = col;
        this.score = score;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Move)) return false;
        Move move = (Move) o;
        return row == move.row && col == move.col;
    }

    @Override
    public int hashCode() {
        return Objects.hash(row, col);
    }

    @Override
    public String toString() {
        return "Move{" + "col=" + col + ", row=" + row + '}';
    }

    public int getRow() {
        return row;
    }

    public void setRow(int row) {
        this.row = row;
    }

    public int getCol() {
        return col;
    }

    public void setCol(int col) {
        this.col = col;
    }

    public int getScore() {
        return score;
    }

    public void setScore(int score) {
        this.score = score;
    }
}
