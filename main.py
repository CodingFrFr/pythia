import numpy as np
import sys
import os
import random
import time

from ui import ConsoleUI
from tokenizer import SimpleTokenizer
from engine import DeltaOptimizer
from model import Pythia

def main():
    Console = ConsoleUI() # sets up ConsoleUI [NOT required]
    Console.boot_sequence(4, 0.3) # [NOT required]
    
    try:
        with open('input.txt', 'r', encoding='utf-8') as f: full_text = f.read() * 10
    except FileNotFoundError:
        print("Error: Missing input.txt")
        return
    tokenizer = SimpleTokenizer(full_text)
    data_ids = np.array(tokenizer.encode(full_text))
    
    # Config (Lightweight)
    BATCH_SIZE = 16
    BLOCK_SIZE = 64
    D_MODEL = 48
    N_LAYER = 3
    STEPS = 500

    # Init Model & Optimizer
    model = Pythia(tokenizer.vocab_size, D_MODEL, N_LAYER, BLOCK_SIZE)
    optimizer = DeltaOptimizer(model.collect_params(), lr=0.005)

    def get_batch():
        ix = np.random.randint(0, len(data_ids) - BLOCK_SIZE, BATCH_SIZE)
        x = np.stack([data_ids[i:i+BLOCK_SIZE] for i in ix])
        y = np.stack([data_ids[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x, y

    def get_lr(step):
        # Cosine Decay
        return 0.005 * 0.5 * (1 + np.cos(np.pi * step / STEPS))

    time.sleep(0.5)
    Console.clear()
    start_time = time.time()
    print(f"\nTraining Pythia on {len(data_ids)} tokens...\n")
    for step in range(STEPS):
        xb, yb = get_batch()
        loss = model.train_step(xb, yb)
        time.sleep(0.001) # Letting the server breathe [only if you're not running locally]
        # Estimating remaining time
        elapsed = time.time() - start_time
        s_per_step = elapsed / (step + 1)
        remaining = (STEPS - step) * s_per_step
        if step % 100 == 0 or step == STEPS - 1:
            print(f"Step {step}/{STEPS} | Loss: {loss:.4f} | ETA: {remaining/60:.1f}m")
    
    total_time = (time.time() - start_time) / 60
    print(f"\n[SUCCESS] Training Complete in {total_time:.1f} minutes.")
    
    # Chat Mode
    Console.clear()
    print("Pythia is ready. Type '/clear' or 'exit'.")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if user_input.lower() == '/clear':
                Console.clear()
                continue

            idx = np.array([tokenizer.encode(user_input)])
            print("Pythia: ", end="")
            
            # Generate with temperature for creativity
            out_ids = model.generate(idx, 60, temperature=0.8)
            
            response_text = tokenizer.decode(out_ids[0][len(idx[0]):])
            Console.type_writer(response_text, speed=0.02)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
