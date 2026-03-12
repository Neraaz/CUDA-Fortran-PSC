import glob
import os
import re

# matches files ending with _number_number_number
tag_regex = re.compile(r".*_\d+_\d+_\d+.*")

def safe_delete(pattern):
    for f in glob.glob(pattern):
        if tag_regex.match(f):
            try:
                os.remove(f)
                print("Deleted:", f)
            except IsADirectoryError:
                pass

def main():
    print("Cleaning generated batch run files...\n")

    # executables
    safe_delete("heat2d_*")

    # slurm + compile scripts
    safe_delete("cpu_*.sh")
    safe_delete("compile_*.sh")

    # profiler outputs
    #safe_delete("nsys_*")

    # slurm logs
    #for f in glob.glob("slurm-*.out"):
    #    os.remove(f)
    #    print("Deleted:", f)

    # generated C file (always safe)
    if os.path.exists("heat2d_cpu.c"):
        os.remove("heat2d_cpu.c")
        print("Deleted: heat2d_cpu.c")

    print("\nTemplates preserved ✅")

if __name__ == "__main__":
    main()
