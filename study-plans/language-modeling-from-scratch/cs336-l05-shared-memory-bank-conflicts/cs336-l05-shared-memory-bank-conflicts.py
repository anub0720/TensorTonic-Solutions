import numpy as np

def bank_conflict_analysis(byte_addresses, num_banks=32, bank_width_bytes=4):
    # Ensure byte_addresses is a numpy array of int64
    addresses = np.asarray(byte_addresses, dtype=np.int64)
    
    # 1. Compute bank IDs for each lane
    bank_ids = (addresses // bank_width_bytes) % num_banks
    
    # 2. Count distinct addresses per bank to determine conflict degree
    # Map each bank to its set of unique addresses
    bank_to_addresses = {}
    for addr, bank in zip(addresses, bank_ids):
        if bank not in bank_to_addresses:
            bank_to_addresses[bank] = set()
        bank_to_addresses[bank].add(addr)
        
    # Each bank's conflict degree is the number of distinct addresses requested from it
    bank_conflict_counts = {bank: len(addrs) for bank, addrs in bank_to_addresses.items()}
    
    # 3. Map back to each lane based on its bank ID
    conflict_degree = np.array([bank_conflict_counts[bank] for bank in bank_ids], dtype=np.int64)
    
    return {
        "bank_ids": bank_ids.astype(np.int64),
        "conflict_degree": conflict_degree
    }