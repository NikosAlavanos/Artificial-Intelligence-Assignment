package org.example.first_order;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class KnowledgeBaseImporter {
    private Resolver resolver = new Resolver();

    public void importFile(String filename) {
        try (InputStream inputStream = org.example.propositional.KnowledgeBase.class.getClassLoader().getResourceAsStream(filename);
             BufferedReader br = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
            String line;

            while ((line = br.readLine()) != null) {
                System.out.println("Processing line: " + line);

                List<Predicate> predicateList = new ArrayList<>();

                String [] tokens = line.split("∨");

                for (String token : tokens) {
                    token = token.trim();

                    boolean isNegated = token.charAt(0) == '¬';

                    if (isNegated) {
                        token = token.replace("¬", "");
                    }

                    int i = token.indexOf('(');
                    int j = token.indexOf(')');

                    String predicateName = token.substring(0, i);

                    String termsStr = token.substring(i+1,j);

                    List<Term> termList = new ArrayList<>();

                    for (String termName : termsStr.split(",")) {
                        termName = termName.trim();

                        boolean variable = Character.isLowerCase(termName.charAt(0));

                        termList.add(new Term(termName, variable));
                    }

                    Predicate predicate = new Predicate(predicateName, termList, isNegated);

                    predicateList.add(predicate);
                }

                Clause c = new Clause(predicateList);

                resolver.addClause(c);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void addClause(Clause clause1) {
        resolver.addClause(clause1);
    }

    public boolean resolve(Clause negatedGoal) {
        return resolver.resolve(negatedGoal);
    }
}
