package org.example.first_order;

import java.util.*;

class Clause {
    List<Predicate> predicates;

    Clause(List<Predicate> predicates) {
        this.predicates = predicates;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Clause other = (Clause) obj;
        return Objects.equals(new HashSet<>(predicates), new HashSet<>(other.predicates)); // Compare sets of predicates
    }

    @Override
    public int hashCode() {
        return Objects.hash(new HashSet<>(predicates));
    }

    @Override
    public String toString() {
        return predicates.toString();
    }
}
