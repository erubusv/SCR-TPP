import numpy as np
import heapq
from collections import defaultdict

class ComplexRule:
    # Multiple source rules occur within a time window -> target event after a delay
    def __init__(self, rule_id, sources, target, window, delay_mu, delay_std, base_prob=1.0):
        """
        :param rule_id: Distinct ID of the rule
        :param sources: List of source event types
        :param target: Target event type
        :param window: Size of the time window that the sources must occur within
        :param delay_mu: Mean delay before the target event occurs
        :param delay_std: Standard deviation of the delay
        :param base_prob: Base probability of the rule triggering
        """
        self.rule_id = rule_id
        self.sources = set(sources)
        self.target = target
        self.window = window
        self.delay_mu = delay_mu
        self.delay_std = delay_std
        self.base_prob = base_prob

    def check_logic(self, history, current_time, current_type):
        if current_type not in self.sources:
            return False

        window_start = current_time - self.window
        
        for src in self.sources:
            if src == current_type:
                continue

            if src not in history or not history[src]:
                return False
            
            if history[src][-1] < window_start:
                return False

        return True
    

def generate_base_events(time_horizon, base_intensities):
    events = []
    current_t = 0.0

    for type_id, rate in base_intensities.items():
        current_t =0.0
        while True:
            if rate <= 0: 
                break

            dt = np.random.exponential(scale=1.0 / rate)
            current_t += dt
            if current_t > time_horizon:
                break

            events.append((current_t, type_id, None))

        events.sort(key=lambda x: x[0])

    return events


def generate_complex_data(rules, interactions, num_samples, time_horizon, base_intensities, max_len=1024):
    """
    :param rules: List of ComplexRule objects defining the logic and interactions
    :param interactions: List of interaction dicts defining how rules affect each other in shape of
                        [{'src': rule_id, 'tgt': rule_id, 'factor': float, 'duration': float}]
                        - Factor < 1.0: Inhibition (0.0=Hard Block, 0.5=Soft Inhibition)
                        - Factor > 1.0: Excitation 
    :param num_samples: Number of sequences to generate
    :param seq_len: Length of each sequence
    :param base_intensities: Dict of base intensities for each event type {type_id: intensity}
    """
    data = []
    
    interaction_map = defaultdict(list)
    for inter in interactions:
        interaction_map[inter['src']].append(inter)

    for _ in range(num_samples):
        # Priority Queue: (time, event_type, source_rule_id)
        # source_rule_id가 None이면 랜덤 발생(Base), 아니면 규칙에 의해 발생
        event_queue = []
        
        # Basic events
        base_events = generate_base_events(time_horizon, base_intensities)
    
        for e in base_events:
            heapq.heappush(event_queue, e)

        final_sequence = []
        history = defaultdict(list) 
        active_modifiers = defaultdict(list)

        # Simulation Loop
        while event_queue:
            curr_t, curr_k, src_rule_id = heapq.heappop(event_queue)
            
            if curr_t > time_horizon or len(final_sequence) >= max_len:
                break
            
            # Float point precision check
            if history[curr_k] and abs(history[curr_k][-1] - curr_t) < 1e-6:
                continue

            final_sequence.append((curr_t, curr_k))
            history[curr_k].append(curr_t)

            # Rule Evaluation
            for rule in rules:
                if rule.check_logic(history, curr_t, curr_k):
                    current_factor = 1.0
                    valid_mods = []
                    # Check interaction modifiers
                    if rule.rule_id in active_modifiers:
                        for expiry, factor in active_modifiers[rule.rule_id]:
                            if expiry > curr_t: 
                                current_factor *= factor
                                valid_mods.append((expiry, factor))

                        active_modifiers[rule.rule_id] = valid_mods
                    
                    final_prob = rule.base_prob * current_factor
                    final_prob = max(0.0, min(1.0, final_prob))
                    
                    # Rule Trigger Check
                    if np.random.rand() < final_prob:
                        delay = np.random.normal(rule.delay_mu, rule.delay_std)
                        new_t = curr_t + delay
                        
                        if new_t <= time_horizon:
                            heapq.heappush(event_queue, (new_t, rule.target, rule.rule_id))
                            
                            if rule.rule_id in interaction_map:
                                for eff in interaction_map[rule.rule_id]:
                                    target_id = eff['tgt']
                                    expiry = curr_t + eff['duration']
                                    factor = eff['factor']
                                    active_modifiers[target_id].append((expiry, factor))

        final_sequence.sort(key=lambda x: x[0])
        if final_sequence:
            times, events = zip(*[(t, k) for t, k in final_sequence])
            data.append({
                'time': list(times),
                'event': list(events)
            })

    return data