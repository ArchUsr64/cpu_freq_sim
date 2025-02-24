import random

# Power scaling factor
ALPHA = 2.5
TASK_TYPE = ["CPU_BOUND", "IO_BOUND", "MIXED", "BACKGROUND"]

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
        cpu_burst_time = random.randint(50, 100)
        io_interrupts = random.randint(1, 5)
    elif task_type == "IO_BOUND":
        cpu_burst_time = random.randint(10, 30)
        io_interrupts = random.randint(20, 50)
    elif task_type == "MIXED":
        cpu_burst_time = random.randint(30, 60)
        io_interrupts = random.randint(10, 30)
    elif task_type == "BACKGROUND":
        cpu_burst_time = random.randint(5, 15)
        io_interrupts = random.randint(1, 5)

    return Task(task_id, cpu_burst_time, io_interrupts, task_type)

print(generate_task(0))
