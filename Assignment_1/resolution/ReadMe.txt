Forward Chaining & Resolution-Based Logical Reasoning

This project provides a practical implementation of two core reasoning techniques for AI systems: Forward Chaining for Horn clauses in first-order predicate logic and Resolution-Based Reasoning for propositional logic. It enables users to load a knowledge base, ask queries, and determine whether the queries can logically follow from the provided information.

---

Features

Forward Chaining:
1. Logical Inference:
   - Processes Horn clauses in first-order logic, deriving conclusions through step-by-step application of rules.
   - Handles constants and variables, using unification to match terms and resolve logical relationships.
2. User Interaction:
   - Accepts user queries via a simple console interface.
   - Evaluates whether a query can be logically derived from the loaded knowledge base.

Resolution-Based Reasoning:
1. Propositional Logic:
   - Employs the resolution method to work with clauses in Conjunctive Normal Form (CNF).
2. Derivation through Contradiction:
   - Combines complementary literals from different clauses to produce new clauses.
   - Detects contradictions in the knowledge base to confirm whether a query is logically valid.

---

How to Run

Setup:
1. Use a Java-compatible IDE, such as IntelliJ IDEA or Eclipse.
2. Ensure JDK version 17 or higher is installed on your system.

Execution:
1. Open the project in your IDE and build it.
2. Run the program, and follow the instructions displayed on the console:
   - Load a knowledge base by specifying the file name.
   - Submit a query to test whether it can be logically derived.

---

Instructions

For Forward Chaining:
1. Create a text file containing Horn clauses in first-order logic. For example:
   ```
   Owns(John, Dog) ∨ ¬Loves(John, Dog)
   Loves(John, Dog)
   ```
2. Run the program and input the file name when prompted.
3. Submit a query in the format of a predicate (e.g., Loves(John, x) or Loves(John, Dog)).
4. The program will process the knowledge base and determine if the query is derivable.

For Resolution-Based Reasoning:
1. Prepare a text file with clauses in CNF format. For example:
   ```
   A V B
   ¬A
   ```
2. Run the program and provide the file name.
3. Input a query as a literal (e.g., A or !A).
4. The system will attempt to prove the query using the resolution method and return the result.

---

Dependencies

- Java Development Kit (JDK): Version 17 or higher.
- Text Files: Knowledge bases must be in plain text format.

This system is a reliable and interactive tool for logical reasoning, combining robust inference mechanisms with a user-friendly interface.