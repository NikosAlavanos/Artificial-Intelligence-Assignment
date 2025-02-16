package org.example.propositional;

import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String filename = "testcase_2.txt";

        KnowledgeBase kb = new KnowledgeBase();

        kb.importFile(filename);

        while (true) {
            System.out.print("Type a siple literal aa query (q to exit): ");
            String line = scanner.nextLine().trim();
//            String line = "P12";
            if (line == null) {
                break;
            }

            line = line.trim();

            if (line.equalsIgnoreCase("quit") || line.equalsIgnoreCase("q")) {
                break;
            }

            boolean swap_result = false;

            if (line.startsWith("!")) {
                line = line.substring(1);

                swap_result = true;
            }

            Literal query = new Literal(line, true);

            boolean result = kb.resolution(query);

            if (swap_result){
                result = !result;
            }

            System.out.println("Result is " + result);
        }
    }
}
