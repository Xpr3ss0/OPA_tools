import numpy as np
from refractiveindex import RefractiveIndexMaterial
from pathlib  import Path

db = Path.home() / ".refractiveindex.info-database" / "data" / "specs" / "ohara" / "optical"

print(db)

materials = sorted(p.stem for p in db.glob("*.yml"))

print(f"{len(materials)} materials found.")
print(materials[:20])

for mat in materials:
    print(f"{mat}")
print(db.parts[-2].upper())

