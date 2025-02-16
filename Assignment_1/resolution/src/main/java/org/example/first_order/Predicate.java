package org.example.first_order;

import java.util.*;

class Predicate {
    String name;
    List<Term> terms;
    boolean isNegated;

    Predicate(String name, List<Term> terms, boolean isNegated) {
        this.name = name;
        this.terms = terms;
        this.isNegated = isNegated;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Predicate other = (Predicate) obj;
        return isNegated == other.isNegated && name.equals(other.name) && terms.equals(other.terms);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, terms, isNegated);
    }

    @Override
    public String toString() {
        return (isNegated ? "~" : "") + name + "(" + terms + ")";
    }
	
	boolean isNegation(Predicate other) {
        return this.name.equals(other.name) && this.isNegated != other.isNegated;
    }
}
