package org.example.propositional;

import java.util.ArrayList;

public class CNFClause {
    private ArrayList<CNFSubClause> subClauses = new ArrayList<>();

    public ArrayList<CNFSubClause> getSubClauses() {
        return this.subClauses;
    }

    public boolean contains(CNFSubClause c) {
        for (int i = 0; i < this.subClauses.size(); i++) {
            if (this.subClauses.get(i).getLiterals().equals(c.getLiterals())) return true;
        }
        return false;
    }
}
