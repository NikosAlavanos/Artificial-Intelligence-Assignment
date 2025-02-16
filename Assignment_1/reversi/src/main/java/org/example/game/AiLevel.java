package org.example.game;

public enum AiLevel {
    RANDOM, // select a random move (uniformly)
    HILL_CLIMBING, // select the move with max value from the table
    MINIMAX, // Minimax
    MINIMAX_ALPHA_BETA // Minimax
}
