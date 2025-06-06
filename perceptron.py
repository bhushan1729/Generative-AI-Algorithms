'''
P <- Set of points with label 1
N <-Set of points with label -1
Initiailize W = [w1, w2,...wn] randomly:
while !convergence do:
    pick random point, say X, from P U N:
    if x belongs to P and w1*x1 + w2*x2 < 0 then
        w = w + alpha
        end
    if x belongs to N and w1*x1 + w2*x2 >= 0 then
        w = w - alpha
        end
end
'''

import numpy as np
import random

class Perceptron:
    def __init__(self, P, N):
        self.P = P  # Positive examples (should yield dot(w, x) > 0)
        self.N = N  # Negative examples (should yield dot(w, x) < 0)
        self.n = len(P[0])  # Dimension

    def calculate_weight(self):        
        # Initialize weight vector and learning rate
        np.random.seed(42)
        w = np.random.uniform(-1, 1, size=self.n)
        alpha = np.ones(self.n) * 0.1
        iter = 0

        while True:
            prev_w = w.copy()
            updated = False

            for x in self.P:
                if np.dot(w, x) <= 0:  # Misclassified positive
                    w = w + alpha      # Move toward the point
                    updated = True

            for x in self.N:
                if np.dot(w, x) >= 0:  # Misclassified negative
                    w = w - alpha      # Move away from the point
                    updated = True

            iter += 1
            print(f"Iteration {iter}: {w}")

            # Check for convergence
            if not updated or np.array_equal(w, prev_w):
                print("No update made. Converged w:", w)
                break

#Example
if __name__ == "__main__":
    P = [(1, 1, 2), (2, 2, 3)]  # Positive class
    N = [(1, 1, 1), (1, 1, 2)]  # Negative class

    perceptron = Perceptron(P, N)
    perceptron.calculate_weight()