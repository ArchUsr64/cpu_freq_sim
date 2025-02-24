import random
import numpy as np

# Power scaling factor
ALPHA = 2.5
# Sorted from high priority to low
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

class CPU:
    def __init__(self, freqs: list[int], algorithm, alpha=ALPHA):
        self.freqs = freqs
        self.current_freq = max(freqs)
        self.alpha = alpha
        self.algorithm = algorithm

    def adjust_frequency(self, task: Task):
        self.current_freq = self.algorithm(task, self.freqs)

    def task_time(self, task: Task) -> float:
        return (task.remaining_time / self.current_freq)

    def power_consumption(self, time: float) -> float:
        return (self.current_freq ** self.alpha) * time

class CPUAlgorithm:
    # Different CPU frequency adjustment strategies.

    def static_frequency(task, freqs):
        return max(freqs)

    def linear_interpolation(task, freqs):
        io_ratio = (task.io_interrupts - IO_RANGE[0]) / (IO_RANGE[1] - IO_RANGE[0])
        return int(np.interp(io_ratio, [0, 1], [min(freqs), max(freqs)]))

    def threshold_based(task, freqs):
        if task.io_interrupts > IO_RANGE[1] * 0.75:
            return min(freqs)
        elif task.io_interrupts > IO_RANGE[1] * 0.5:
            return freqs[1]
        elif task.io_interrupts > IO_RANGE[1] * 0.25:
            return freqs[2]
        else:
            return max(freqs)

    def proportional_scaling(task, freqs):
        return int(min(freqs) + (max(freqs) - min(freqs)) * (1 - (task.io_interrupts / IO_RANGE[1])))

    def random_selection(task, freqs):
        return random.choice(freqs)

cpus = [CPU(FREQS, CPUAlgorithm.linear_interpolation), CPU(FREQS, CPUAlgorithm.static_frequency)]

tasks = [generate_task(i) for i in range(1_000)]
tasks.sort(key = lambda x: TASK_TYPE.index(x.task_type))

for cpu in cpus:
    total_time = 0
    total_power = 0

    for task in tasks:
        cpu.adjust_frequency(task)
        time_taken = cpu.task_time(task)
        power_used = cpu.power_consumption(time_taken)

        total_time += time_taken
        total_power += power_used

        # print(f"Executing {task}, CPU Freq: {cpu.current_freq} MHz, Time: {time_taken:.3f}s, Power: {power_used:.3f} units")

    print(f"Total time: {total_time:.1f}s, Total power: {total_power // 1000}kUnits")

