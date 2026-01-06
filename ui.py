import sys
import os
import random
import time

class ConsoleUI:
    def __init__(self, speed=0.2):
        self.prefixes = [
                "[KERNEL]", "[BLAS]", "[PYTHIA]", "[TENSOR]", 
                "[CUDA]", "[MEM]", "[NET]", "[LOAD]"
        ]
            
        self.verbs = [
                "Hydrating", "Allocating", "Quantizing", "Fusing", 
                "Sharding", "Broadcasting", "Compiling", "Injecting",
                "Pruning", "Verifying"
        ]
            
        self.nouns = [
                "Rotary Positional Embeddings", "FlashAttention-v2 kernels",
                "deduplicated Pile indices", "GPT-NeoX transformer blocks",
                "Float32 accumulator buffers", "cosine learning rate scheduler",
                "stochastic depth masks", "KV-cache slots",
                "dense synaptic weights", "non-linearities"
        ]
            
        self.statuses = [
                "... OK", "... DONE", "... LOCKED", "... SYNCHRONIZED",
                "... [12ms]", "... [0.004s]", "... BYPASSED"
        ]
        self.speed = speed
        self.banner = """
    .      *       .       .   *      .
  *    P  Y  T  H  I  A    v1.5   *
    .   *   _______   .   *   .
       ____/_______\\____
      /__________________\\    
    .   *   .       .   .   *   .
        """

    def generate_line(self):
        """Constructs a single random boot line."""
        if random.random() > 0.3:
            return f"{random.choice(self.prefixes)} {random.choice(self.verbs)} {random.choice(self.nouns)}{random.choice(self.statuses)}"
        else:
            val = random.randint(1024, 65535)
            return f"{random.choice(self.prefixes)} {random.choice(self.nouns)} check: {val} blocks verified."

    def clear(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(self.banner)

    def type_writer(self, text, speed=0.1, color_code=None):
        if color_code: sys.stdout.write(color_code)
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(speed + random.uniform(-0.005, 0.005))
        if color_code: sys.stdout.write('\033[0m')
        print()

    def boot_sequence(self, lines=4, speed=0.1):
        self.clear()
        for _ in range(int(lines)):
            line = self.generate_line()
            if "Quantizing" in line or "Compiling" in line:
                time.sleep(speed * 4)
            print(line)
            time.sleep(random.uniform(speed * 0.5, speed * 1.5))
        time.sleep(0.3)
        self.clear()
        print("\n[READY] Model inference state: ACTIVE\n")
        time.sleep(0.6)