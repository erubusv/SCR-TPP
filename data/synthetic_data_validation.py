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
    
    total_seqs = len(dataset)
    print(f"Total Sequences: {total_seqs}")

    # Logic rule verification
    print("\n[1. Logic Rule Verification]")
    
    for rule in rules:
        target_counts = 0
        explained_counts = 0
        
        max_lookback = rule.window + rule.delay_mu + 3 * rule.delay_std
        min_lookback = rule.delay_mu - 3 * rule.delay_std
        
        for seq in dataset:
            times = np.array(seq['time'])
            events = np.array(seq['event'])
            
            target_indices = np.where(events == rule.target)[0]
            
            for t_idx in target_indices:
                t_target = times[t_idx]
                target_counts += 1

                # Check for source events in the lookback window
                search_mask = (times >= t_target - max_lookback) & (times < t_target - min_lookback)
                recent_events = set(events[search_mask])

                if rule.sources.issubset(recent_events):
                    explained_counts += 1

        if target_counts > 0:
            ratio = explained_counts / target_counts * 100
            print(f"  Rule {rule.rule_id} ({rule.sources} -> {rule.target}):")
            print(f"    - Total Targets: {target_counts}")
            print(f"    - Explained ratio: {ratio:.1f} %")

        else:
            print(f"  Rule {rule.rule_id}: No target events found.")

    # Interaction verification
    print("\n[2. Interaction Verification]")
    
    interaction_map = defaultdict(list)
    for inter in interactions:
        interaction_map[inter['src']].append(inter)

    for src_rule_id, effects in interaction_map.items():
        # Find triggering rule
        src_rule = next((r for r in rules if r.rule_id == src_rule_id), None)
        if not src_rule:
            continue
        
        src_event_type = src_rule.target
        
        for eff in effects:
            target_rule_id = eff['tgt']
            tgt_rule = next((r for r in rules if r.rule_id == target_rule_id), None)
            if not tgt_rule: 
                continue
            
            target_event_type = tgt_rule.target
            duration = eff['duration']
            factor = eff['factor']
            type_str = "Inhibition" if factor < 1.0 else "Excitation"
            
            total_time_global = 0.0
            total_time_active = 0.0
            
            count_global = 0
            count_active = 0
            
            for seq in dataset:
                times = np.array(seq['time'])
                events = np.array(seq['event'])

                if len(times) == 0:
                    continue
                
                total_time_global += times[-1]
                count_global += np.sum(events == target_event_type)

                src_indices = np.where(events == src_event_type)[0]
                raw_intervals = []
                for s_idx in src_indices:
                    t_start = times[s_idx]
                    t_end = min(t_start + duration, times[-1])
                    if t_end > t_start:
                        raw_intervals.append((t_start, t_end))

                merged_len, merged_intervals = merge_intervals(raw_intervals)
                total_time_active += merged_len

                target_indices = np.where(events == target_event_type)[0]
                for t_idx in target_indices:
                    t_ev = times[t_idx]
                    if is_in_interval(t_ev, merged_intervals):
                        count_active += 1

            total_time_non_active = total_time_global - total_time_active
            count_non_active = count_global - count_active

            density_active = count_active / total_time_active if total_time_active > 1e-6 else 0.0
            density_non_active = count_non_active / total_time_non_active if total_time_non_active > 1e-6 else 0.0
            
            print(f"  Interaction (Rule {src_rule_id} -> Rule {target_rule_id}): {type_str} (Factor {factor})")
            print(f"    - Non-Active Rate (Background): {density_non_active:.4f} Hz")
            print(f"    - Active Rate (Interaction):    {density_active:.4f} Hz")
            
            passed = False
            if factor < 1.0: # Inhibition
                if density_active < density_non_active: passed = True
            else: # Excitation
                if density_active > density_non_active: passed = True
                
            status = "VERIFIED" if passed else "WARNING"
            print(f"    {status} (Active vs Non-Active)")


if __name__ == "__main__":
    rules = [
        ComplexRule(rule_id=1, sources={1, 3}, target=5, window=10.0, delay_mu=5.0, delay_std=1.0, base_prob=0.8),
        ComplexRule(rule_id=2, sources={3, 4}, target=6, window=8.0, delay_mu=3.0, delay_std=0.5, base_prob=0.5),
        ComplexRule(rule_id=3, sources={2}, target=7, window=5.0, delay_mu=2.5, delay_std=0.5, base_prob=0.6),
    ]
    interactions = [
        {'src': 3, 'tgt': 2, 'duration': 15.0, 'factor': 0.2},  # Inhibition
        {'src': 1, 'tgt': 2, 'duration': 10.0, 'factor': 2.0},  # Excitation
    ]

    data = generate_complex_data(rules, interactions, num_samples=500, time_horizon=2000.0, base_intensities={1: 1.5, 2: 0.8, 3: 0.7, 4: 1.0, 5: 0.05, 6: 0.05, 7: 0.05})
    validate_generated_data(data, rules, interactions)