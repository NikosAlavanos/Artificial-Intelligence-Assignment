package org.example.first_order;

import java.util.*;

class Resolver {
    List<Clause> knowledgeBase = new ArrayList<>();

    void addClause(Clause clause) {
        knowledgeBase.add(clause);
    }

    boolean resolve(Clause goal) {
		Queue<Clause> toResolve = new LinkedList<>(knowledgeBase);
		Set<Clause> seenClauses = new HashSet<>(knowledgeBase); // Track processed clauses
		toResolve.add(goal);
	
		while (!toResolve.isEmpty()) {
			Clause clause = toResolve.poll();
	
			for (Clause kbClause : knowledgeBase) {
				Clause resolvedClause = resolveClauses(clause, kbClause);
	
				if (resolvedClause != null && resolvedClause.predicates.isEmpty()) {
                    System.out.print("Resolving " + clause + " with " + kbClause);
					// Contradiction found: Empty clause

                    Clause c2 = resolveClauses(clause, kbClause);

					return true;
				}

                if (resolvedClause != null && !seenClauses.contains(resolvedClause)) {
                    System.out.print("Resolving " + clause + " with " + kbClause);
					// Add only new clauses
					seenClauses.add(resolvedClause);
					toResolve.add(resolvedClause);

                    System.out.println("\n\tNew sentence: " + resolvedClause);
                }
			}
		}
		return false; // No contradiction, goal is not derivable
	}


    private Clause resolveClauses(Clause c1, Clause c2) {
        for (Predicate p1 : c1.predicates) {
            for (Predicate p2 : c2.predicates) {
                if (p1.isNegation(p2)) {
                    Unifier unifier = new Unifier();
                    boolean canUnify = true;
                    for (int i = 0; i < p1.terms.size(); i++) {
                        if (!unifier.unify(p1.terms.get(i), p2.terms.get(i))) {
                            canUnify = false;
                            break;
                        }
                    }
                    if (canUnify) {
                        Set<Predicate> newPredicates = new HashSet<>();
                        newPredicates.addAll(c1.predicates);
                        newPredicates.addAll(c2.predicates);
                        newPredicates.remove(p1);
                        newPredicates.remove(p2);

                        return new Clause(new ArrayList<>(newPredicates));
                    }
                }
            }
        }
        return null;
    }
}