package org.example.propositional;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;

public class KnowledgeBase extends CNFClause {

    public void importFile(String filename) {
        try (InputStream inputStream = KnowledgeBase.class.getClassLoader().getResourceAsStream(filename);
             BufferedReader br = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
            String line;

            while ((line = br.readLine()) != null) {
                System.out.println("Processing line: " + line);

                CNFSubClause clause = new CNFSubClause();

                String [] tokens = line.split("V");

                for (String t : tokens) {
                    t = t.trim();

                    if (t.startsWith("!")) {
                        t = t.substring(1);
                        Literal l = new Literal(t, true);
                        clause.getLiterals().add(l);
                    } else{
                        Literal l = new Literal(t, false);
                        clause.getLiterals().add(l);
                    }
                }

                this.getSubClauses().add(clause);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public boolean resolution(Literal a) {
        CNFClause KB = this;
        CNFClause clauses = new CNFClause();
        clauses.getSubClauses().addAll(KB.getSubClauses());

        Literal notA = new Literal(a.getName(), !a.isNegated());
        CNFSubClause aClause = new CNFSubClause();
        aClause.getLiterals().add(notA);
        clauses.getSubClauses().add(aClause);

        while (true) {
            ArrayList<CNFSubClause> newS = new ArrayList<CNFSubClause>();
            ArrayList<CNFSubClause> subClauses = clauses.getSubClauses();

            for (int i = 0; i < subClauses.size(); i++) {
                CNFSubClause Ci = subClauses.get(i);

                for (int j = i + 1; j < subClauses.size(); j++) {
                    CNFSubClause Cj = subClauses.get(j);

                    ArrayList<CNFSubClause> resolvents = CNFSubClause.resolve(Ci, Cj); // resolution.

                    for (CNFSubClause resolvent : resolvents) {
                        //...and if an empty subclause has been generated we have reached contradiction; and the literal has been proved
                        if (resolvent.isEmpty()) {
                            Ci.print();
                            System.out.print(" , ");
                            Cj.print();
                            System.out.print("     ⊨     ");
                            System.out.println("Empty sub-clause!");
                            return true;
                        }

                        //All clauses produced that don't exist already are added
                        if (!newS.contains(resolvent) && !clauses.contains(resolvent)) {
                            Ci.print();
                            System.out.print(" , ");
                            Cj.print();
                            System.out.print("     ⊨     ");
                            resolvent.print();
                            newS.add(resolvent);
                            System.out.println();
                        }
                    }
                }
            }

            boolean newClausesFound = false;
            //Check if any new clauses were produced in this loop
            if (newS.size() > 0) {
                clauses.getSubClauses().addAll(newS);
                newClausesFound = true;
            }


            //If not, then Knowledge Base does not logically infer the literal we wanted to prove
            if (!newClausesFound) {
                System.out.println("New clauses were not found.");
                return false;
            }
        }
    }
}
