from ortools.sat.python import cp_model

model = cp_model.CpModel()

# Create variables representing the seat position (0-5) for each person
amir = model.NewIntVar(0, 5, 'amir')
bella = model.NewIntVar(0, 5, 'bella')
charles = model.NewIntVar(0, 5, 'charles')
diana = model.NewIntVar(0, 5, 'diana')
ethan = model.NewIntVar(0, 5, 'ethan')
farah = model.NewIntVar(0, 5, 'farah')

# All people must sit in different seats
model.AddAllDifferent([amir, bella, charles, diana, ethan, farah])

# Constraint: Bella must sit to the left of Farah
model.Add(bella < farah)

# Constraint: Charles and Diana must sit next to each other
model.Add((charles == diana - 1) or (charles == diana + 1))

# Constraint: Amir must not sit at either end (seats 0 or 5)
model.Add(amir != 0)
model.Add(amir != 5)

# Constraint: Ethan must sit in the middle section (seats 2 or 3)
model.Add(ethan >= 2)
model.Add(ethan <= 3)

# Constraint: Diana must not sit at either end (seats 0 or 5)
model.Add(diana != 0)
model.Add(diana != 5)

# Solve the model
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Print results
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    # 1. Display the resulting assignment
    print("Solution found:")
    assignments = {
        "Amir": solver.Value(amir),
        "Bella": solver.Value(bella),
        "Charles": solver.Value(charles),
        "Diana": solver.Value(diana),
        "Ethan": solver.Value(ethan),
        "Farah": solver.Value(farah)
    }
    
    # Display the panel arrangement
    print("\nPanel seating arrangement (seats 0-5):")
    seats = [""] * 6
    for person, seat in assignments.items():
        seats[seat] = person
    for i, person in enumerate(seats):
        print(f"Seat {i}: {person}")
    
    # 2. Verify that all constraints are satisfied
    print("\nConstraint verification:")
    print(f"1. All people are in different seats: {len(set(assignments.values())) == 6}")
    print(f"2. Bella sits to the left of Farah: {assignments['Bella'] < assignments['Farah']}")
    print(f"3. Charles and Diana sit next to each other: {abs(assignments['Charles'] - assignments['Diana']) == 1}")
    print(f"4. Amir is not at either end: {assignments['Amir'] != 0 and assignments['Amir'] != 5}")
    print(f"5. Ethan sits in middle section: {2 <= assignments['Ethan'] <= 3}")
    print(f"6. Diana is not at either end: {assignments['Diana'] != 0 and assignments['Diana'] != 5}")
    
    # 3. Comment on solver status
    if status == cp_model.OPTIMAL:
        print("\nSolver status: OPTIMAL - The solver found the optimal solution.")
    else:
        print("\nSolver status: FEASIBLE - The solver found a feasible solution but cannot prove it's optimal.")
    
else:
    print("No solution found.")