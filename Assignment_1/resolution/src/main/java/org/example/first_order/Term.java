package org.example.first_order;

class Term {
    String name;
    boolean isVariable;

    Term(String name, boolean isVariable) {
        this.name = name;
        this.isVariable = isVariable;
    }

    @Override
    public String toString() {
        return name;
    }
}