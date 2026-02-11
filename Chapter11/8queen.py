"""
8-Queens Problem solved with a Genetic Algorithm.
Each individual is a list of 8 integers (0-7), where index = column
and value = row of the queen in that column.
"""

import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ── GA Parameters ──
POP_SIZE = 200
MUTATION_RATE = 0.3
GENERATIONS = 1000
BOARD_SIZE = 8

# ── Fitness: count non-attacking pairs (max = 28) ──
def fitness(individual):
    non_attacks = 0
    for i in range(BOARD_SIZE):
        for j in range(i + 1, BOARD_SIZE):
            if individual[i] != individual[j] and \
               abs(individual[i] - individual[j]) != abs(i - j):
                non_attacks += 1
    return non_attacks  # 28 = perfect solution

# ── Create a random individual ──
def create_individual():
    return [random.randint(0, BOARD_SIZE - 1) for _ in range(BOARD_SIZE)]

# ── Tournament selection ──
def select(population):
    a, b = random.sample(population, 2)
    return a if fitness(a) >= fitness(b) else b

# ── Single-point crossover ──
def crossover(parent1, parent2):
    point = random.randint(1, BOARD_SIZE - 1)
    child = parent1[:point] + parent2[point:]
    return child

# ── Mutate one gene ──
def mutate(individual):
    if random.random() < MUTATION_RATE:
        i = random.randint(0, BOARD_SIZE - 1)
        individual[i] = random.randint(0, BOARD_SIZE - 1)
    return individual

# ── Plot the chessboard with queens ──
def plot_board(solution, generation):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw chessboard squares
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            color = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
            ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor=color))

    # Place queens
    for col, row in enumerate(solution):
        ax.text(col + 0.5, row + 0.5, "Q", fontsize=24, fontweight="bold",
                ha="center", va="center", color="#222222")

    ax.set_xlim(0, BOARD_SIZE)
    ax.set_ylim(0, BOARD_SIZE)
    ax.set_xticks(range(BOARD_SIZE))
    ax.set_yticks(range(BOARD_SIZE))
    ax.set_xticklabels([chr(ord('a') + i) for i in range(BOARD_SIZE)])
    ax.set_yticklabels([str(i + 1) for i in range(BOARD_SIZE)])
    ax.set_aspect("equal")
    ax.set_title(f"8-Queens Solution  (found at generation {generation})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

# ── Main GA loop ──
if __name__ == "__main__":
    random.seed(42)

    population = [create_individual() for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        # Check for a perfect solution
        best = max(population, key=fitness)
        best_fit = fitness(best)

        if gen % 50 == 0 or best_fit == 28:
            print(f"Generation {gen:4d}  |  Best fitness = {best_fit}/28")

        if best_fit == 28:
            print(f"\nSolution found at generation {gen}: {best}")
            plot_board(best, gen)
            break

        # Build next generation
        new_population = [best]            # elitism: keep the best
        while len(new_population) < POP_SIZE:
            p1, p2 = select(population), select(population)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
    else:
        # Ran out of generations
        best = max(population, key=fitness)
        print(f"\nBest after {GENERATIONS} generations: {best}  "
              f"(fitness {fitness(best)}/28)")
        plot_board(best, GENERATIONS)
