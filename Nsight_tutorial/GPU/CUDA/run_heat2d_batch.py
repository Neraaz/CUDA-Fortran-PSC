import subprocess

CONFIG_FILE = "configs.txt"

def run(cmd):
    print(">>", cmd)
    subprocess.run(cmd, shell=True, check=True)

def write_file(name, content):
    with open(name, "w") as f:
        f.write(content)

def load_template(fname):
    with open(fname) as f:
        return f.read()

heat_template = load_template("heat2d_template.cu")
cpu_template = load_template("cpu_template.sh")
compile_template = load_template("compile_template.sh")

with open(CONFIG_FILE) as f:
    for line in f:
        if line.strip().startswith("#") or not line.strip():
            continue

        nx, ny, nt = map(int, line.split())

        tag = f"{nx}_{ny}_{nt}"
        exe = f"heat2d_{tag}"
        jobname = f"heat-{tag}"
        nsys_out = f"nsys_{tag}"
        cpu_script = f"cpu_{tag}.sh"
        compile_script = f"compile_{tag}.sh"

        print(f"\n===== CASE {tag} =====")

        # ---- 1) generate C file ----
        code = heat_template.replace("NX_PLACEHOLDER", str(nx))
        code = code.replace("NY_PLACEHOLDER", str(ny))
        code = code.replace("NT_PLACEHOLDER", str(nt))
        write_file("heat2d_gpu.cu", code)

        # ---- 2) generate compile script ----
        comp = compile_template.replace("EXECUTABLE", exe)
        write_file(compile_script, comp)

        # ---- 3) compile ----
        run(f"bash {compile_script}")
        #run(f"chmod +x {exe}")

        # ---- 4) generate SLURM script ----
        cpu = cpu_template.replace("JOBNAME", jobname)
        cpu = cpu.replace("NSYS_OUT", nsys_out)
        cpu = cpu.replace("EXECUTABLE", f"./{exe}")
        write_file(cpu_script, cpu)

        # ---- 5) submit job ----
        run(f"sbatch {cpu_script}")
