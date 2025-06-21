1. model return mask of shape (example, x, 2) and (example, y, 2). more expressive would be to return mask of shape (example, x, y, 2).
2. model assumes size of test output is bound by max in the training data (counter example, double grid size task). Try using a global max size of 30.
tradeoff between compute and accuracy.
3. multitensor system is conceptually elegant, but not efficient for computation. Reimplement with a single flattened tensor?
4. share down function, special treatment for x and y axes just average over the top left corner but enumerate over all possible x and y offsets in train.py.
better to not have special treatment?