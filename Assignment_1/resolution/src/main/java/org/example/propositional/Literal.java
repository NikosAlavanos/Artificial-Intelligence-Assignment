package org.example.propositional;

public class Literal {
    private String name;
    private boolean isNegated;

    public Literal(String name, boolean neg) {
        this.name = name;
        this.isNegated = neg;
    }

    public boolean isNegated() {
        return this.isNegated;
    }

    public String getName() {
        return this.name;
    }

    public void print() {
        if (this.isNegated) {
            System.out.print("¬" + this.name);
        } else {
            System.out.print(this.name);
        }
    }

    @Override
    public boolean equals(Object obj) {
        Literal l = (Literal) obj;
        if (l.getName().equals(this.name) && l.isNegated() == this.isNegated) return true;
        else return false;
    }

    @Override
    public int hashCode() {
        if (this.isNegated) return this.name.hashCode() + 1;
        else return this.name.hashCode();
    }
}
