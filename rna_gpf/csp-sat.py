from ortools.sat.python import cp_model

model = cp_model.CpModel()

seq = "AUGCG"

A1 = model.NewIntVar(0, 5, "A1")
U2 = model.NewIntVar(0, 5, "U2")
G3 = model.NewIntVar(0, 5, "G3")
C4 = model.NewIntVar(0, 5, "C4")
G5 = model.NewIntVar(0, 5, "G5")

