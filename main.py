import random

# Power scaling factor
ALPHA = 2.5
TASK_TYPE = ["CPU_BOUND", "IO_BOUND", "MIXED", "BACKGROUND"]
# Frequencies in MHz
FREQS = [500, 800, 1000, 1500]
IO_RANGE = [1, 50]
CPU_BURST_RANGE = [5, 100]

class Task:
    def __init__(self, id, cpu_burst_time: int, io_interrupts: int, task_type):
        self.id = id
        self.remaining_time = cpu_burst_time
        self.io_interrupts = io_interrupts
        self.task_type = task_type

    def __repr__(self):
        return f"Task(id={self.id}, type={self.task_type}, CPU={self.remaining_time}ms, IO={self.io_interrupts})"

def generate_task(task_id):
    task_type = random.choice(TASK_TYPE)

    if task_type == "CPU_BOUND":
        cpu_burst_time = random.randint(CPU_BURST_RANGE[0] // 2, CPU_BURST_RANGE[1])
        io_interrupts = random.randint(IO_RANGE[0], IO_RANGE[1] // 4)
    elif task_type == "IO_BOUND":
        cpu_burst_time = random.randint(CPU_BURST_RANGE[0], CPU_BURST_RANGE[1] // 4)
        io_interrupts = random.randint(IO_RANGE[0] // 2, IO_RANGE[1])
    elif task_type == "MIXED":
        cpu_burst_time = random.randint(CPU_BURST_RANGE[0], CPU_BURST_RANGE[1] // 2)
        io_interrupts = random.randint(IO_RANGE[0], IO_RANGE[1] // 2)
    elif task_type == "BACKGROUND":
        cpu_burst_time = random.randint(CPU_BURST_RANGE[0], CPU_BURST_RANGE[1] // 4)
        io_interrupts = random.randint(IO_RANGE[0], IO_RANGE[1] // 4)

    return Task(task_id, cpu_burst_time, io_interrupts, task_type)
