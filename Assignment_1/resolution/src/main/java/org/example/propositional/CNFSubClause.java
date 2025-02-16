package org.example.propositional;

import java.util.ArrayList;
import java.util.HashSet;

public class CNFSubClause {
    private HashSet<Literal> literals = new HashSet<Literal>();

    public HashSet<Literal> getLiterals() {
        return this.literals;
    }

    public boolean isEmpty() {
        return this.literals.isEmpty();
    }

    public void print() {
//        System.out.println("**************");
        int counter = 0;
        for (Literal l : this.literals) {
            l.print();

            if (counter < this.literals.size()-1) {
                System.out.print(" ∨ ");
            }

            counter++;
        }
//        System.out.println("**************");
    }

    /* Applies resolution on two CNFSubClauses
     * The resulting clause will contain all the literals of both CNFSubclauses
     * except the pair of literals that are a negation of each other.
     */
    public static ArrayList<CNFSubClause> resolve(CNFSubClause C1, CNFSubClause C2) {
        ArrayList<CNFSubClause> newSubClause = new ArrayList<CNFSubClause>();

        //The iterator goes through all Literals of the first clause
        for (Literal l : C1.getLiterals()) {
            Literal m = new Literal(l.getName(), !l.isNegated()); // negated l

            //If the second clause contains the negation of a Literal in the first clause
            if (C2.getLiterals().contains(m)) {
                //We construct a new clause that contains all the literals of both CNFSubclauses...
                HashSet<Literal> C1exceptL = new HashSet<Literal>(C1.getLiterals());
                C1exceptL.remove(l); // remove l

                HashSet<Literal> C2exceptM = new HashSet<Literal>(C2.getLiterals());
                C2exceptM.remove(m); // remove m

                //Normally we have to remove duplicates of the same literal; the new clause must not contain the same literal more than once
                //But since we use HashSet only one copy of a literal will be contained anyway
                CNFSubClause result = new CNFSubClause();

                result.getLiterals().addAll(C1exceptL);
                result.getLiterals().addAll(C2exceptM);

                newSubClause.add(result);
            }

        }//The loop runs for all literals, producing a different new clause for each different pair of literals that negate each other

        return newSubClause;
    }

    @Override
    public boolean equals(Object obj) {
        CNFSubClause c = (CNFSubClause) obj;

        if (c.getLiterals().size() != this.getLiterals().size()) return false;

        for (Literal lit : c.getLiterals()) {
            if (!this.getLiterals().contains(lit)) return false;
        }

        return true;
    }

    @Override
    public int hashCode() {
        int code = 0;
        for (Literal l : this.literals) {
            code += l.hashCode();
        }

        return code;
    }
}
