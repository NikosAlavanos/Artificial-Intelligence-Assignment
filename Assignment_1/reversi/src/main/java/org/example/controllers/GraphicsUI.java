package org.example.controllers;

import org.example.board.Color;
import org.example.board.Disc;
import org.example.config.Configuration;
import org.example.game.AiLevel;
import org.example.game.Game;
import org.example.game.Move;
import org.example.game.Player;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.List;
import java.util.Random;

public class GraphicsUI extends JFrame implements GameUI {
    private static final int BOARD_SIZE = 8;
    private final Game game;
    private CellButton[][] cells;
    private JLabel statusLabel;
    private JLabel scoreLabel;

    public GraphicsUI(Game game) {
        this.game = game;
        setTitle("Reversi Game");
        setSize(600, 700);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        setLocationRelativeTo(null);
        setResizable(false);
    }

    private void showGameSetupDialog() {
        JPanel setupPanel = new JPanel(new BorderLayout(10, 10));
        setupPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JPanel modePanel = new JPanel(new GridLayout(4, 1, 5, 5));
        modePanel.setBorder(BorderFactory.createTitledBorder("Select Game Mode"));
        ButtonGroup modeGroup = new ButtonGroup();
        JRadioButton humanVsHuman = new JRadioButton("Human vs Human", true);
        JRadioButton humanVsAi = new JRadioButton("Human vs AI");
        JRadioButton aiVsHuman = new JRadioButton("AI vs Human");
        JRadioButton aiVsAi = new JRadioButton("AI vs AI");

        modeGroup.add(humanVsHuman);
        modeGroup.add(humanVsAi);
        modeGroup.add(aiVsHuman);
        modeGroup.add(aiVsAi);

        modePanel.add(humanVsHuman);
        modePanel.add(humanVsAi);
        modePanel.add(aiVsHuman);
        modePanel.add(aiVsAi);

        JPanel aiPanel = new JPanel(new GridLayout(5, 2, 10, 10));
        aiPanel.setBorder(BorderFactory.createTitledBorder("AI Levels"));

        JLabel blackAiLabel = new JLabel("Black AI Level:");
        JComboBox<String> blackAiLevelSelector = new JComboBox<>(new String[]{"Random", "Hill Climbing", "Minimax", "Minimax - Pruning"});

        JLabel blackAiDepthLabel = new JLabel("Black AI Depth:");
        JTextField blackAiDepthField = new JTextField();
        blackAiDepthField.setEnabled(false);

        JLabel whiteAiLabel = new JLabel("White AI Level:");
        JComboBox<String> whiteAiLevelSelector = new JComboBox<>(new String[]{"Random", "Hill Climbing", "Minimax", "Minimax - Pruning"});

        JLabel whiteAiDepthLabel = new JLabel("White AI Depth:");
        JTextField whiteAiDepthField = new JTextField();
        whiteAiDepthField.setEnabled(false);

        aiPanel.add(blackAiLabel);
        aiPanel.add(blackAiLevelSelector);
        aiPanel.add(blackAiDepthLabel);
        aiPanel.add(blackAiDepthField);
        aiPanel.add(whiteAiLabel);
        aiPanel.add(whiteAiLevelSelector);
        aiPanel.add(whiteAiDepthLabel);
        aiPanel.add(whiteAiDepthField);

        blackAiLevelSelector.addActionListener(e -> {
            boolean isMinimax = blackAiLevelSelector.getSelectedItem().toString().contains("Minimax");
            blackAiDepthField.setEnabled(isMinimax);
        });

        whiteAiLevelSelector.addActionListener(e -> {
            boolean isMinimax = whiteAiLevelSelector.getSelectedItem().toString().contains("Minimax");
            whiteAiDepthField.setEnabled(isMinimax);
        });

        humanVsAi.addActionListener(e -> {
            blackAiLevelSelector.setEnabled(false);
            whiteAiLevelSelector.setEnabled(true);
            blackAiDepthField.setEnabled(false);
            whiteAiDepthField.setEnabled(whiteAiLevelSelector.getSelectedItem().toString().contains("Minimax"));
        });

        aiVsHuman.addActionListener(e -> {
            blackAiLevelSelector.setEnabled(true);
            whiteAiLevelSelector.setEnabled(false);
            blackAiDepthField.setEnabled(blackAiLevelSelector.getSelectedItem().toString().contains("Minimax"));
            whiteAiDepthField.setEnabled(false);
        });

        aiVsAi.addActionListener(e -> {
            blackAiLevelSelector.setEnabled(true);
            whiteAiLevelSelector.setEnabled(true);
            blackAiDepthField.setEnabled(blackAiLevelSelector.getSelectedItem().toString().contains("Minimax"));
            whiteAiDepthField.setEnabled(whiteAiLevelSelector.getSelectedItem().toString().contains("Minimax"));
        });

        humanVsHuman.addActionListener(e -> {
            blackAiLevelSelector.setEnabled(false);
            whiteAiLevelSelector.setEnabled(false);
            blackAiDepthField.setEnabled(false);
            whiteAiDepthField.setEnabled(false);
        });

        setupPanel.add(modePanel, BorderLayout.NORTH);
        setupPanel.add(aiPanel, BorderLayout.CENTER);

        int result = JOptionPane.showConfirmDialog(
                this,
                setupPanel,
                "Game Setup",
                JOptionPane.OK_CANCEL_OPTION,
                JOptionPane.PLAIN_MESSAGE
        );

        if (result == JOptionPane.OK_OPTION) {
            Player black;
            Player white;

            if (humanVsHuman.isSelected()) {
                black = new Player(1, Color.BLACK, false, null, 0);
                white = new Player(2, Color.WHITE, false, null, 0);
            } else if (humanVsAi.isSelected()) {
                black = new Player(1, Color.BLACK, false, null, 0);
                white = new Player(2, Color.WHITE, true, parseAiLevel(whiteAiLevelSelector.getSelectedItem().toString()), parseAiDepth(whiteAiDepthField));
            } else if (aiVsHuman.isSelected()) {
                black = new Player(1, Color.BLACK, true, parseAiLevel(blackAiLevelSelector.getSelectedItem().toString()), parseAiDepth(blackAiDepthField));
                white = new Player(2, Color.WHITE, false, null, 0);
            } else {
                black = new Player(1, Color.BLACK, true, parseAiLevel(blackAiLevelSelector.getSelectedItem().toString()), parseAiDepth(blackAiDepthField));
                white = new Player(2, Color.WHITE, true, parseAiLevel(whiteAiLevelSelector.getSelectedItem().toString()), parseAiDepth(whiteAiDepthField));
            }

            game.start(black, white);
            initializeGameBoard();

            if (black.isAi() || white.isAi()) {
                handleAiTurns();
            }
        } else {
            System.exit(0);
        }
    }

    private int parseAiDepth(JTextField depthField) {
        try {
            return Integer.parseInt(depthField.getText().trim());
        } catch (NumberFormatException e) {
            return Configuration.MAX_DEPTH;
        }
    }


    private void handleAiTurns() {
        new Thread(() -> {
            while (!game.isOver()) {
                Player currentPlayer = game.getState().activePlayer == Color.BLACK ? game.getState().black : game.getState().white;

                if (currentPlayer.isAi()) {
                    disableBoard();
                    try {
                        Thread.sleep(1000);
                        Move aiMove = currentPlayer.thinkMove(game.getState());
                        game.playMove(aiMove);

                        SwingUtilities.invokeLater(() -> {
                            updateBoard();
                            Player nextPlayer = game.getState().activePlayer == Color.BLACK ? game.getState().black : game.getState().white;
                            if (!nextPlayer.isAi()) {
                                enableBoard();
                            }
                        });
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                } else {
                    enableBoard();
                    break;
                }
            }
        }).start();
    }

    private AiLevel parseAiLevel(String aiLevel) {
        if (aiLevel.equalsIgnoreCase("Random")) {
            return AiLevel.RANDOM;
        }
        if (aiLevel.equalsIgnoreCase("Hill climbing")) {
            return AiLevel.HILL_CLIMBING;
        }
        if (aiLevel.equalsIgnoreCase("Minimax")) {
            return AiLevel.MINIMAX;
        }
        if (aiLevel.equalsIgnoreCase("Minimax - Pruning")) {
            return AiLevel.MINIMAX_ALPHA_BETA;
        }

        throw new IllegalArgumentException("Invalid AI LEVEL");
    }

    private void initializeGameBoard() {
        JPanel boardPanel = new JPanel(new GridLayout(BOARD_SIZE, BOARD_SIZE));
        cells = new CellButton[BOARD_SIZE][BOARD_SIZE];

        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                cells[row][col] = new CellButton(row, col);
                cells[row][col].setBackground(new java.awt.Color(0, 102, 0));
                cells[row][col].setPreferredSize(new Dimension(60, 60));
                cells[row][col].setBorder(BorderFactory.createLineBorder(java.awt.Color.BLACK));
                cells[row][col].addActionListener(new CellClickListener(row, col));
                boardPanel.add(cells[row][col]);
            }
        }

        statusLabel = new JLabel("Turn: Black", SwingConstants.CENTER);
        statusLabel.setFont(new Font("Arial", Font.BOLD, 16));

        scoreLabel = new JLabel("Score - White: 0 | Black: 0", SwingConstants.CENTER);
        scoreLabel.setFont(new Font("Arial", Font.PLAIN, 14));

        add(statusLabel, BorderLayout.NORTH);
        add(boardPanel, BorderLayout.CENTER);
        add(scoreLabel, BorderLayout.SOUTH);

        updateBoard();
        setVisible(true);
    }

    private void updateBoard() {
        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                cells[row][col].setBackground(new java.awt.Color(0, 102, 0));
            }
        }

        int whiteScore = 0;
        int blackScore = 0;
        List<Move> validMoves = game.getState().getValidMoves(game.getState().activePlayer);

        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                Disc disc = game.getState().discs[row][col];
                CellButton cell = cells[row][col];

                if (disc != null) {
                    cell.setDiscColor(mapToAwtColor(disc.getColor()));
                    if (disc.getColor() == Color.WHITE) whiteScore++;
                    else blackScore++;
                } else {
                    cell.setDiscColor(null);

                    if (validMoves.contains(new Move(row, col))) {
                        cell.setBackground(java.awt.Color.YELLOW);
                    }
                }
            }
        }

        scoreLabel.setText("Score - White: " + whiteScore + " | Black: " + blackScore);
        statusLabel.setText("Turn: " + (game.getState().activePlayer == Color.WHITE ? "White" : "Black"));

        if (game.getState().isGameOver()) {
            endGame(whiteScore, blackScore);
        }
    }

    private java.awt.Color mapToAwtColor(Color boardColor) {
        return boardColor == Color.WHITE ? java.awt.Color.WHITE : java.awt.Color.BLACK;
    }

    private void disableBoard() {
        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                cells[row][col].setEnabled(false);
            }
        }
    }

    private void enableBoard() {
        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                cells[row][col].setEnabled(true);
            }
        }
    }

    private void endGame(int whiteScore, int blackScore) {
        String winner;
        String imagePath;

        if (whiteScore > blackScore) {
            winner = "White";
            imagePath = "src/Images/Lost";
        } else if (blackScore > whiteScore) {
            winner = "Black";
            imagePath = "src/Images/Win";
        } else {
            winner = "Nobody (It's a tie)";
            imagePath = "src/Images/Draw";
        }

        try {
            String randomImage = getRandomImageFromFolder(imagePath);
            if (randomImage != null) {
                ImageIcon icon = new ImageIcon(randomImage);
                JOptionPane.showMessageDialog(this,
                        "Game Over! Winner: " + winner + "\nFinal Score - White: " + whiteScore + " | Black: " + blackScore,
                        "Game Over",
                        JOptionPane.INFORMATION_MESSAGE,
                        icon);
            } else {
                JOptionPane.showMessageDialog(this,
                        "Game Over! Winner: " + winner + "\nFinal Score - White: " + whiteScore + " | Black: " + blackScore);
            }
        } catch (Exception e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(this,
                    "Error loading winner image.\nGame Over! Winner: " + winner +
                            "\nFinal Score - White: " + whiteScore + " | Black: " + blackScore);
        }
    }

    private String getRandomImageFromFolder(String folderPath) {
        File folder = new File(folderPath);
        File[] imageFiles = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".png"));

        if (imageFiles != null && imageFiles.length > 0) {
            Random random = new Random();
            File selectedImage = imageFiles[random.nextInt(imageFiles.length)];
            return selectedImage.getAbsolutePath();
        }
        return null;
    }

    @Override
    public void run() {
        showGameSetupDialog();
    }

    private class CellClickListener implements ActionListener {
        private final int row;
        private final int col;

        public CellClickListener(int row, int col) {
            this.row = row;
            this.col = col;
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            Move move = new Move(row, col);
            if (game.getState().getValidMoves(game.getState().activePlayer).contains(move)) {
                game.playMove(move);
                updateBoard();
                handleAiTurns();
            } else {
                JOptionPane.showMessageDialog(GraphicsUI.this, "Invalid move! Try again.", "Invalid Move", JOptionPane.WARNING_MESSAGE);
            }
        }
    }

    private class CellButton extends JButton {
        private java.awt.Color discColor;

        public CellButton(int row, int col) {
            this.discColor = null;
        }

        public void setDiscColor(java.awt.Color color) {
            this.discColor = color;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (discColor != null) {
                g.setColor(discColor);
                g.fillOval(10, 10, getWidth() - 20, getHeight() - 20);
            }
        }
    }
}
