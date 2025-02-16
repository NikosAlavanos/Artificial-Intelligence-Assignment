package org.example;

import org.example.controllers.GameUI;
import org.example.controllers.GraphicsUI;
import org.example.controllers.CommandLineUI;
import org.example.game.Game;

public class Main {
    public static void main(String[] args) {
        Game game = new Game();

        GameUI controller;

        if (args.length == 0 || args[0].equalsIgnoreCase("--ui")) {
            controller = new GraphicsUI(game);
        } else {
            controller = new CommandLineUI(game);
        }

        controller.run();
    }
}