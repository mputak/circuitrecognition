import os

work_dir = os.listdir(".\\validation_res")
for dir in work_dir:
    os.startfile(f".\\validation_res\\{dir}")