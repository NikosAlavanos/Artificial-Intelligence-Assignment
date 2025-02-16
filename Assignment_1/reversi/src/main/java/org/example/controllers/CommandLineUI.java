package org.example.controllers;

import org.example.board.Color;
import org.example.config.Configuration;
import org.example.game.AiLevel;
import org.example.game.Game;
import org.example.game.Move;
import org.example.game.Player;

import java.util.Scanner;

public class CommandLineUI implements GameUI {
    private final Game game;
    private final Scanner scanner = new Scanner(System.in);
    private int maxdepth;


    public CommandLineUI(Game game) {
        this.game = game;
    }

    public void printMenu() {
        System.out.println("-----------------------------------");
        System.out.println("                 M E N U ");
        System.out.println("-----------------------------------");

        System.out.println("    1. Black Human vs White Human");
        System.out.println("    2. Black Human vs White AI");
        System.out.println("    3. Black AI vs White Human");
        System.out.println("    4. Black AI vs White AI");
        System.out.println("    0. Exit");
    }

    public int readInput() {
        while (true) {
            System.out.print("Enter your choice [0-4]: ");

            String data = scanner.nextLine();

            try {
                int choice = Integer.parseInt(data);

                if (choice == 1 || choice == 2 || choice == 3 || choice == 4  || choice == 0) {
                    return choice;
                } else {
                    System.out.println("Invalid choice");
                }
            } catch (Exception ex) {
                System.out.println("Invalid choice. Please enter 1,2,3,4  or 0");
            }
        }
    }

    private int readInputMaxDepth(Color color) {
        while (true) {
            System.out.print("Please enter the search depth for the " +  color + " AI: ");

            String data = scanner.nextLine();

            return maxdepth = Integer.parseInt(data);
        }
    }
    public AiLevel selectAiLevel() {
        while (true) {
            System.out.println("Select AI level:");
            System.out.println("    1. Random");
            System.out.println("    2. Hill Climbing");
            System.out.println("    3. Minimax");
            System.out.println("    4. Minimax - Pruning");

            System.out.print("Enter your choice [1-4]: ");
            String data = scanner.nextLine();
            try {
                int choice = Integer.parseInt(data);
                switch (choice) {
                    case 1: return AiLevel.RANDOM;
                    case 2: return AiLevel.HILL_CLIMBING;
                    case 3: return AiLevel.MINIMAX;
                    case 4: return AiLevel.MINIMAX_ALPHA_BETA;
                    default: System.out.println("Invalid choice");
                }
            } catch (Exception ex) {
                System.out.println("Invalid choice. Please enter 1 or 2");
            }
        }
    }

    public void run() {
        while (true) {
            printMenu();
            int choice = (Configuration.DEBUG_MODE) ? 2 : readInput();

            if (choice == 0) {
                break;
            }
            AiLevel blackAiLevel = null;
            AiLevel whiteAiLevel = null;

            if (choice == 3 || choice == 4) {
                System.out.println("Black AI selected.");
                blackAiLevel = selectAiLevel();
                if (blackAiLevel == AiLevel.MINIMAX || blackAiLevel== AiLevel.MINIMAX_ALPHA_BETA){
                    maxdepth = readInputMaxDepth(Color.BLACK);
                    System.out.println(Color.BLACK + " AI depth is: " + maxdepth);
                }
            }
            if (choice == 2 || choice == 4) {
                System.out.println("White AI selected.");
                whiteAiLevel = (Configuration.DEBUG_MODE) ? AiLevel.MINIMAX : selectAiLevel();
                if (whiteAiLevel == AiLevel.MINIMAX || whiteAiLevel== AiLevel.MINIMAX_ALPHA_BETA){
                    maxdepth = readInputMaxDepth(Color.WHITE);
                    System.out.println(Color.WHITE + " AI depth is: " + maxdepth);
                }
            }

            Player p1 = new Player(1, Color.BLACK, choice == 3 || choice == 4, blackAiLevel, maxdepth);
            Player p2 = new Player(2, Color.WHITE, choice == 2 || choice == 4, whiteAiLevel, maxdepth);

            game.start(p1, p2);

            while (!game.isOver()) {
                game.print();

                Move move = game.selectMove();

                System.out.println("Move selected: " + move);

                game.playMove(move);
            }

            game.print();

            game.showTheWinner();
        }

    }
}
