from ortools.sat.python import cp_model

def solve_bin_packing():
    model = cp_model.CpModel()
    
    # Constants
    weights = [4, 8, 1, 4, 2, 6, 3, 5]
    num_items = len(weights)
    num_bins = 3
    bin_capacity = 12
    
    # Variables: bin_assignment[i] represents which bin (0, 1, or 2) item i is assigned to
    bin_assignment = []
    for i in range(num_items):
        bin_assignment.append(model.NewIntVar(0, num_bins-1, f'item_{i}_bin'))
    
    # Create binary variables to track which items go in which bins (for bin capacity constraints)
    is_in_bin = {}
    for i in range(num_items):
        for b in range(num_bins):
            is_in_bin[(i, b)] = model.NewBoolVar(f'item_{i}_in_bin_{b}')
            # Link binary variables to bin assignment
            model.Add(bin_assignment[i] == b).OnlyEnforceIf(is_in_bin[(i, b)])
            model.Add(bin_assignment[i] != b).OnlyEnforceIf(is_in_bin[(i, b)].Not())
    
    # Constraint: Each bin's total weight must not exceed capacity
    for b in range(num_bins):
        model.Add(sum(weights[i] * is_in_bin[(i, b)] for i in range(num_items)) <= bin_capacity)
    
    # Constraint: Items 0 and 1 must be in different bins
    model.Add(bin_assignment[0] != bin_assignment[1])
    
    # Constraint: Items 2 and 3 must be in the same bin
    model.Add(bin_assignment[2] == bin_assignment[3])
    
    # Constraint: Items 4 and 5 must be in different bins
    model.Add(bin_assignment[4] != bin_assignment[5])
    
    # Constraint: Items 6 and 7 must be in the same bin
    model.Add(bin_assignment[6] == bin_assignment[7])
    
    # Constraint: Items 6 and 7 must not share a bin with item 1
    model.Add(bin_assignment[6] != bin_assignment[1])
    
    # Constraint: Item 5 is not allowed in bin 0
    model.Add(bin_assignment[5] != 0)
    
    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    # Print results
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution found:")
        
        # Track which items are in which bins
        bins = [[] for _ in range(num_bins)]
        bin_weights = [0] * num_bins
        
        for i in range(num_items):
            b = solver.Value(bin_assignment[i])
            bins[b].append(i)
            bin_weights[b] += weights[i]
        
        # Display bin contents
        print("\nBin assignments:")
        for b in range(num_bins):
            print(f"Bin {b}: Items {bins[b]} (weights: {[weights[i] for i in bins[b]]}, total: {bin_weights[b]})")
        
        # Verify constraints
        print("\nConstraint verification:")
        print(f"1. All bin capacities respected: {all(w <= bin_capacity for w in bin_weights)}")
        print(f"2. Items 0 and 1 in different bins: {solver.Value(bin_assignment[0]) != solver.Value(bin_assignment[1])}")
        print(f"3. Items 2 and 3 in same bin: {solver.Value(bin_assignment[2]) == solver.Value(bin_assignment[3])}")
        print(f"4. Items 4 and 5 in different bins: {solver.Value(bin_assignment[4]) != solver.Value(bin_assignment[5])}")
        print(f"5. Items 6 and 7 in same bin: {solver.Value(bin_assignment[6]) == solver.Value(bin_assignment[7])}")
        print(f"6. Items 6/7 not with item 1: {solver.Value(bin_assignment[6]) != solver.Value(bin_assignment[1])}")
        print(f"7. Item 5 not in bin 0: {solver.Value(bin_assignment[5]) != 0}")
        
        if status == cp_model.OPTIMAL:
            print("\nSolver status: OPTIMAL - Found the optimal solution.")
        else:
            print("\nSolver status: FEASIBLE - Found a solution but cannot prove it's optimal.")
    else:
        print("No solution found.")

if __name__ == "__main__":
    solve_bin_packing()