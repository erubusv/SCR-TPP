import numpy as np
from collections import defaultdict
from synthetic import generate_complex_data, ComplexRule

def merge_intervals(intervals):
    if not intervals:
        return 0.0, []
        
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    current_start, current_end = intervals[0]
    
    for next_start, next_end in intervals[1:]:
        if next_start < current_end: 
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged.append((current_start, current_end))
    
    total_len = sum(end - start for start, end in merged)
    return total_len, merged


def is_in_interval(t, intervals):
    for start, end in intervals:
        if start < t <= end:
            return True
    return False


def validate_generated_data(dataset, rules, interactions):
    # Examine the generated synthetic data to follow the defined rules and interactions.
    print(f"Total Sequences: {len(dataset)}")

    # Logic rule verification
    print("\n[1. Logic Rule Verification]")

    for rule in rules:
        target_counts = 0
        valid_logic_counts = 0
    
        for seq in dataset:
            times = np.array(seq['time'])
            events = np.array(seq['event'])
            labels = np.array(seq['label'])
            
            triggered_indices = np.where(labels == rule.rule_id)[0]
            
            for t_idx in triggered_indices:
                target_counts += 1
                t_target = times[t_idx]
                
                all_sources_found = True
                
                for src_type, config in rule.source_configs.items():
                    ideal_src_time = t_target - config['mu']
                    tolerance = 3 * config['std']
                    
                    search_start = ideal_src_time - tolerance
                    search_end = ideal_src_time + tolerance
                    
                    mask = (times >= search_start) & (times <= search_end) & (events == src_type)
                    
                    if not np.any(mask):
                        all_sources_found = False
                        break
                
                if all_sources_found:
                    valid_logic_counts += 1

        if target_counts > 0:
            ratio = valid_logic_counts / target_counts * 100
            print(f"  Rule {rule.rule_id} (Target {rule.target}):")
            print(f"    - Triggered: {target_counts} times")
            print(f"    - Logic Validated: {ratio:.1f}% (Matched source timings)")
        else:
            print(f"  Rule {rule.rule_id}: Not triggered.")


    print("\n[2. Interaction Verification]")
    
    rules_map = {r.rule_id: r for r in rules}

    for i, inter in enumerate(interactions):
        srcs = inter.get('sources', inter.get('src'))
        if not isinstance(srcs, list): srcs = [srcs]
        
        target_rule_id = inter['tgt']
        factor = inter['factor']
        beta = inter['beta']
        
        target_rule = rules_map.get(target_rule_id)
        if not target_rule:
            print(f"  Warning: Target Rule {target_rule_id} not found.")
            continue
            
        target_delay = min(cfg['mu'] for cfg in target_rule.source_configs.values())
        
        effective_duration = 5.0 / (beta + 1e-9)
        
        type_str = "Excitation (+)" if factor > 1.0 else "Inhibition (-)"
        
        total_time_active = 0.0
        total_time_global = 0.0
        
        count_active = 0
        count_global = 0
        
        for seq in dataset:
            times = np.array(seq['time'])
            labels = np.array(seq['label'])
            
            if len(times) == 0: continue
            total_time_global += times[-1]
            
            count_global += np.sum(labels == target_rule_id)
            
            src_occurrences = {}
            for src_id in srcs:
                src_occurrences[src_id] = times[labels == src_id]
            
            raw_intervals = []
            
            candidate_times = []
            for t_list in src_occurrences.values():
                candidate_times.extend(t_list)
            candidate_times = sorted(list(set(candidate_times)))
            
            for t_curr in candidate_times:
                all_satisfied = True
                for src_id in srcs:
                    valid_acts = src_occurrences[src_id][src_occurrences[src_id] <= t_curr]
                    if len(valid_acts) == 0:
                        all_satisfied = False
                        break
                    last_t = valid_acts[-1]
                    if (t_curr - last_t) * beta >= 5.0:
                        all_satisfied = False
                        break
                        
                if all_satisfied:
                    start_active = t_curr + target_delay
                    end_active = start_active + effective_duration
                    
                    if start_active < times[-1]:
                        raw_intervals.append((start_active, min(end_active, times[-1])))
            
            merged_len, merged_intervals = merge_intervals(raw_intervals)
            total_time_active += merged_len
            
            target_indices = np.where(labels == target_rule_id)[0]
            for t_idx in target_indices:
                t_ev = times[t_idx]
                if is_in_interval(t_ev, merged_intervals):
                    count_active += 1
                    
        total_time_non_active = max(0, total_time_global - total_time_active)
        count_non_active = count_global - count_active
        
        density_active = count_active / total_time_active if total_time_active > 1e-6 else 0.0
        density_non_active = count_non_active / total_time_non_active if total_time_non_active > 1e-6 else 0.0
        
        ratio = density_active / (density_non_active + 1e-9)
        
        print(f"  Interaction (Sources {srcs} -> Target {target_rule_id}): {type_str} Factor={factor}")
        print(f"    - Shifted Window: +{target_delay:.1f}s (Adjusted for Delay)")
        print(f"    - Background Rate: {density_non_active:.4f} Hz")
        print(f"    - Active Rate:     {density_active:.4f} Hz")
        print(f"    - Observed Ratio:  {ratio:.2f}")
        
        passed = False
        if factor > 1.0: 
            if density_active > density_non_active: passed = True
        else:
            if density_active < density_non_active: passed = True
            
        status = "PASSED" if passed else "WARNING"
        print(f"    -> {status}")


if __name__ == "__main__":
    # Test Setup
    rules = [
        ComplexRule(rule_id=1, 
                    source_configs={1: {'mu': 5.0, 'std': 1.0}, 3: {'mu': 1.0, 'std': 0.2}}, 
                    target=5, base_prob=0.8),

        ComplexRule(rule_id=2, 
                    source_configs={3: {'mu': 3.0, 'std': 0.5}, 4: {'mu': 3.0, 'std': 0.5}}, 
                    target=6, base_prob=0.5),
        
        ComplexRule(rule_id=3, 
                    source_configs={2: {'mu': 2.5, 'std': 0.5}}, 
                    target=7, base_prob=0.6),
    ]

    interactions = [
        {'sources': [3], 'tgt': 1, 'beta': 1.0, 'factor': 0.2}, 
        
        {'sources': [1], 'tgt': 2, 'beta': 0.5, 'factor': 2.0},
        
        {'sources': [1, 3], 'tgt': 2, 'beta': 2.0, 'factor': 5.0}
    ]

    data = generate_complex_data(
        rules, interactions, 
        num_samples=500, 
        time_horizon=1000.0, 
        base_intensities={1: 1.5, 2: 0.8, 3: 0.7, 4: 1.0, 5: 0.05, 6: 0.05, 7: 0.05}
    )
    
    validate_generated_data(data, rules, interactions)