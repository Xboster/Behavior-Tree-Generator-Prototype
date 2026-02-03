# Behavior Tree Generator Prototype

This generates random behavior trees for game AI using a grammar-based system.

At a high level, the generator works in two stages:

1. AST Generation: Recursively expands a set of grammar rules to produce an abstract syntax tree (AST) representing the behavior tree (control nodes and leaves).

2. Converts the AST into actual Behavior Tree objects (Sequence, Selector, Condition, Action)

### Control & Parameters:

Currently the only parameter is max_depth. It limits the maximum levels the tree that it will generate. I'm planning on adding a way for it to parse a grammar file for it to use to generate the trees. Probably in json or something.

### Risk Review & Next Steps:

Currently the behavior trees that it generates are pretty incoherent, I'm planning on addressing this by adding a way to create grammar rules.
