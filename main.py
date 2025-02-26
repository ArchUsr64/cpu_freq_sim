import random
from algorithm import IOAwareCPUAlgorithm
import numpy as np

# Power scaling factor
ALPHA = 2.5
# Sorted from high priority to low
TASK_TYPE = ["CPU_BOUND", "IO_BOUND", "MIXED", "BACKGROUND"]
# Frequencies in MHz
FREQS = [400, 800, 1000, 1200]
IO_RANGE = [10, 500]
CPU_BURST_RANGE = [5, 100]
TASKS_COUNT = 20_000

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
        return task.io_interrupts + self.task_cpu_time(task)

    def task_cpu_time(self, task: Task) -> float:
        return task.remaining_time / (self.current_freq / min(self.freqs))

    def power_consumption(self, task: Task) -> float:
        return (self.current_freq ** self.alpha) * self.task_time(task)

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

io_aware_algorithm = IOAwareCPUAlgorithm(
    # Standard CPU frequency levels (in MHz)
    freqs=FREQS,
    # Power scaling factor
    alpha=ALPHA,
    # Moderate smoothing to react to trends without overfitting
    smoothing_factor=1,
    # Slight penalty for I/O-heavy tasks
    penalty_factor=0.15,
    # Ensures low-I/O tasks still get some CPU
    min_freq_weight=0.5,
    # Biases towards running CPU-bound tasks efficiently
    max_freq_weight=2.75,
    # Adaptive threshold for defining "high" I/O intensity
    io_threshold=sum(IO_RANGE) / 2,
    # Uses last n tasks to smooth trends
    history_size=10,
    # Moderate entropy impact (keeps tuning flexible)
    entropy_weight=0.25,
    # Keeps prediction influence subtle, not overcorrecting
    prediction_factor=0.35,
    # Removes extreme cases from historical data
    outlier_rejection=True,
)


cpus = [
    CPU(FREQS, CPUAlgorithm.static_frequency),
    CPU(FREQS, CPUAlgorithm.linear_interpolation),
    CPU(FREQS, CPUAlgorithm.threshold_based),
    CPU(FREQS, CPUAlgorithm.proportional_scaling),
    CPU(FREQS, CPUAlgorithm.random_selection),
    CPU(FREQS, io_aware_algorithm.adjust_frequency)
]

tasks = [generate_task(i) for i in range(TASKS_COUNT)]
tasks.sort(key = lambda x: TASK_TYPE.index(x.task_type))

for cpu in cpus:
    total_time = 0
    total_power = 0

    for task in tasks:
        cpu.adjust_frequency(task)
        time_taken = cpu.task_time(task)
        power_used = cpu.power_consumption(task)

        total_time += time_taken
        total_power += power_used

        # print(f"{task}, Freq: {cpu.current_freq}MHz, Time: {time_taken:.3f}s")

    print(f"Time: {total_time:.1f}s, Power: {total_power // 1000}kUnits, Efficiency:\t{TASKS_COUNT /(total_time * total_power)}")

