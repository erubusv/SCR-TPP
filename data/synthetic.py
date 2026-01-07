import numpy as np
import heapq
from collections import defaultdict

class ComplexRule:
    # Multiple source rules occur within a time window -> target event after a delay
    def __init__(self, rule_id, source_configs, target, base_prob=1.0):
        """
        :param rule_id: Distinct ID of the rule
        :param source_configs: Dict defining requirements for each source type.
                               Format: { source_type_id: {'mu': float, 'std': float}, ... }
                               - mu: How far back in time the source must occur (Lag)
                               - std: Standard Deviation of the lag
        :param target: Target event type
        :param base_prob: Base probability of the rule triggering
        """
        self.rule_id = rule_id
        self.source_configs = source_configs
        self.sources = set(source_configs.keys())
        self.target = target
        self.base_prob = base_prob

    
    def check_pattern_consensus(self, history, current_time, current_type):
        """
        Check if the current event (current_time, current_type) can trigger the rule
        based on consensus from other source events in history.
        """
        if current_type not in self.sources:
            return None

        my_config = self.source_configs[current_type]
        projected_target_time = current_time + my_config['mu']
        
        matched_times = [current_time]

        for src, config in self.source_configs.items():
            if src == current_type: continue
            
            ideal_src_time = projected_target_time - config['mu']
            tolerance = 3 * config['std']
            
            found = False
            if src in history:
                for past_t in reversed(history[src]):
                    if past_t < ideal_src_time - tolerance:
                        break
                    
                    if abs(past_t - ideal_src_time) <= tolerance:
                        matched_times.append(past_t)
                        found = True
                        break 
            
            if not found:
                return None

        return projected_target_time, matched_times
    

def generate_complex_data(rules, interactions, num_samples, time_horizon, base_intensities, max_len=1024):
    data = []
    
    interactions_by_target = defaultdict(list)
    for inter in interactions:
        srcs = inter.get('sources', inter.get('src'))
        if not isinstance(srcs, list):
            srcs = [srcs]
        
        inter_config = {
            'sources': set(srcs),
            'tgt': inter['tgt'],
            'factor': inter['factor'],
            'beta': inter['beta']
        }
        interactions_by_target[inter['tgt']].append(inter_config)
        
    rules_by_trigger = defaultdict(list)
    for rule in rules:
        for src in rule.sources:
            rules_by_trigger[src].append(rule)

    for _ in range(num_samples):
        event_queue = []

        for type_id, rate in base_intensities.items():
            if rate <= 0: 
                continue

            first_t = np.random.exponential(scale=1.0 / rate)
            if first_t <= time_horizon:
                heapq.heappush(event_queue, (first_t, type_id, 0))

        full_sequence = []
        history = defaultdict(list) 
        rule_last_occurrence = {}

        while event_queue:
            curr_t, curr_k, src_rule_id = heapq.heappop(event_queue)
            
            if curr_t > time_horizon: 
                break
            if len(full_sequence) >= max_len:
                break
            
            if history[curr_k] and abs(history[curr_k][-1] - curr_t) < 1e-6:
                continue

            # Baseline Event Generation
            if src_rule_id == 0:
                rate = base_intensities[curr_k]
                dt = np.random.exponential(scale=1.0 / rate)
                next_t = curr_t + dt
                
                if next_t <= time_horizon:
                    heapq.heappush(event_queue, (next_t, curr_k, 0))

            # Event Commit
            full_sequence.append((curr_t, curr_k, src_rule_id))
            history[curr_k].append(curr_t)

            if src_rule_id > 0:
                rule_last_occurrence[src_rule_id] = curr_t

            # Rule evaluation
            potential_rules = rules_by_trigger[curr_k]
            for rule in potential_rules:
                consensus = rule.check_pattern_consensus(history, curr_t, curr_k)
                
                if consensus:
                    target_proposal_time, matched_times = consensus
                    interaction_effect_sum = 0.0
                    
                    if rule.rule_id in interactions_by_target:
                        
                        for inter in interactions_by_target[rule.rule_id]:
                            required_sources = inter['sources']
                            beta = inter['beta']
                            factor = inter['factor']
                            
                            source_times = []
                            all_satisfied = True
                            
                            for src_id in required_sources:
                                last_t = rule_last_occurrence.get(src_id)
                                
                                if last_t is None:
                                    all_satisfied = False
                                    break
                                
                                dt = curr_t - last_t
                                if dt * beta >= 5.0:
                                    all_satisfied = False
                                    break
                                    
                                source_times.append(last_t)
                            
                            if all_satisfied:
                                t_latest = max(source_times)
                                delay_from_completion = curr_t - t_latest
                                
                                weight = np.exp(-beta * delay_from_completion)
                                interaction_effect_sum += (factor - 1.0) * weight

                    current_factor = max(0.0, 1.0 + interaction_effect_sum)
                    
                    final_prob = rule.base_prob * current_factor
                    final_prob = max(0.0, min(1.0, final_prob))
                    
                    if np.random.rand() < final_prob:
                        jitter = np.random.normal(0, 0.1)
                        new_t = max(curr_t + 0.01, target_proposal_time + jitter)
                        
                        if new_t <= time_horizon:
                            heapq.heappush(event_queue, (new_t, rule.target, rule.rule_id))

        full_sequence.sort(key=lambda x: x[0])
        if full_sequence:
            times, events, labels = zip(*full_sequence)
            data.append({
                'time': list(times),
                'event': list(events),
                'label': list(labels)
            })

    return data