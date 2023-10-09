import os
name="Rishabh"
if not os.path.exists(f'Faces/{name}'):
    with open(f'Faces/{name}','x') as f:
        f.write("Name,Time")    
