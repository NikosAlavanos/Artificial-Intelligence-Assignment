package org.example.first_order;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
//Owns(John, Dog)
public class Main {
    public static Clause parse(String queryString) {
        List<Predicate> predicateList = new ArrayList<>();

        boolean isNegated = true;

        int i = queryString.indexOf('(');
        int j = queryString.indexOf(')');

        String predicateName = queryString.substring(0, i);

        String termsStr = queryString.substring(i+1,j);

        List<Term> termList = new ArrayList<>();

        for (String termName : termsStr.split(",")) {
            termName = termName.trim();

            boolean variable = Character.isLowerCase(termName.charAt(0));

            termList.add(new Term(termName, variable));
        }

        Predicate predicate = new Predicate(predicateName, termList, isNegated);

        predicateList.add(predicate);

        Clause c = new Clause(predicateList);

        return c;
    }

    public static void main(String[] args) {
        KnowledgeBaseImporter kb = new KnowledgeBaseImporter();
        String filename = "fol_testcase_1.txt";

        kb.importFile(filename);

        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print("Type a sentence aa query (q to exit): ");
            String line = scanner.nextLine().trim();

            if (line == null) {
                break;
            }

            line = line.trim();

            if (line.equalsIgnoreCase("quit") || line.equalsIgnoreCase("q")) {
                break;
            }

            Clause negatedGoal = parse(line);

            if (kb.resolve(negatedGoal)) {
                System.out.println("Contradiction found: Goal is derivable.");
            } else {
                System.out.println("No contradiction: Goal is not derivable.");
            }
        }
    }
}