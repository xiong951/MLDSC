import numpy as np
import time
from collections import defaultdict
import random
from ExpGaussMix import ExpGaussMix
from scipy.stats import linregress
from scipy.signal import savgol_filter

# Extract the longest continuous group of ones from each row in the array.
def process_dom_array(dom):
    processed_dom = np.copy(dom)
    first_groups = []

    for i in range(dom.shape[0]):
        row = dom[i]
        one_indices = np.where(row == 1)[0]

        if len(one_indices) == 0:
            first_groups.append([])
            continue

        groups = []
        current_group = [one_indices[0]]

        for idx in one_indices[1:]:
            if idx == current_group[-1] + 1:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
        groups.append(current_group)

        groups.sort(key=lambda g: len(g), reverse=True)

        first_group = groups[0] if groups else []

        if first_group and len(first_group) < 20:
            first_group = []

        first_groups.append(first_group)

        keep_indices = set(first_group)
        if len(one_indices) > 0:
            for idx in one_indices:
                if idx not in keep_indices:
                    processed_dom[i, idx] = 0

    return first_groups

# A reinforcement learning environment to optimize peak boundary selection.
class PeakBoundaryRL:
    # Initialize the reinforcement learning environment parameters and limits.
    def __init__(self, temperature, signal, peak_center, is_connected_peak=False,
                 custom_left=None, custom_right=None, process_type='endo', mode='strict'):
        self.process_type = process_type
        self.mode = mode
        self.temperature = temperature
        self.signal = signal
        self.peak_center = peak_center
        self.n_points = len(temperature)

        range_limit = int(self.n_points * 0.25)
        self.search_limit_left = max(0, peak_center - range_limit)
        self.search_limit_right = min(self.n_points - 1, peak_center + range_limit)

        if self.process_type == 'exo':
            limit_ratio = 0.06
            max_dist = int(self.n_points * limit_ratio)

            force_limit = max(0, self.peak_center - max_dist)

            self.search_limit_left = max(self.search_limit_left, force_limit)

        self.y_span = np.max(signal) - np.min(signal) + 1e-6

        if self.mode == 'strict':
            edge_len = max(10, int(self.n_points * 0.05))
            noise_sample = np.concatenate([signal[:edge_len], signal[-edge_len:]])
            q75, q25 = np.percentile(noise_sample, [75, 25])
            iqr = q75 - q25
            self.noise_level = max(iqr * 1.5, 1e-5 * self.y_span)
        else:
            self.noise_level = 0.02 * self.y_span

        self.global_slope, self.global_intercept = self._estimate_background_trend()

        if self.n_points < 800:
            self.actions = [
                "expand_left", "shrink_left", "expand_right", "shrink_right",
                "confirm"
            ]
        else:
            self.actions = [
                "expand_left", "shrink_left", "expand_right", "shrink_right",
                "expand_both", "shrink_both", "confirm"
            ]

        l, r = self._calculate_derivative_initial_bounds(peak_center, signal)
        width = r - l
        margin = max(30, int(width * 1.5))

        factor = 2.0 if (self.mode == 'loose' and self.process_type == 'endo') else 1.0

        init_l = l - int(margin * factor)
        init_r = r + int(margin * factor)

        self.initial_left = max(self.search_limit_left, init_l)
        self.initial_right = min(self.search_limit_right, init_r)

        if self.initial_left >= self.initial_right:
            self.initial_left = max(self.search_limit_left, peak_center - 10)
            self.initial_right = min(self.search_limit_right, peak_center + 10)

        if self.mode == 'loose_exo_right':
            self.noise_level = 0.03 * self.y_span

    # Estimate the global baseline slope and intercept using the edges of the signal.
    def _estimate_background_trend(self):
        margin_idx = int(self.n_points * 0.10)
        if margin_idx < 5:
            slope = (self.signal[-1] - self.signal[0]) / (self.temperature[-1] - self.temperature[0])
            return slope, self.signal[0] - slope * self.temperature[0]
        x_bg = np.concatenate([self.temperature[:margin_idx], self.temperature[-margin_idx:]])
        y_bg = np.concatenate([self.signal[:margin_idx], self.signal[-margin_idx:]])
        slope, intercept, _, _, _ = linregress(x_bg, y_bg)
        return slope, intercept

    # Calculate initial left and right boundaries based on the signal derivative.
    def _calculate_derivative_initial_bounds(self, center_idx, signal):
        n = len(signal)
        w = max(5, int(n * 0.02))
        smooth = np.convolve(signal, np.ones(w) / w, mode='same')
        grads = np.abs(np.gradient(smooth))
        limit = int(n * 0.25)
        l_search = max(0, center_idx - limit)
        r_search = min(n, center_idx + limit)
        threshold = np.max(grads[l_search:r_search]) * 0.05
        l_bound = max(0, center_idx - 10)
        for i in range(center_idx, 0, -1):
            if grads[i] < threshold:
                l_bound = i
                break
        r_bound = min(n - 1, center_idx + 10)
        for i in range(center_idx, n - 1):
            if grads[i] < threshold:
                r_bound = i
                break
        return l_bound, r_bound

    # Calculate the slope and intercept of the baseline connecting two given points.
    def _get_baseline_params(self, left, right):
        x1, y1 = self.temperature[left], self.signal[left]
        x2, y2 = self.temperature[right], self.signal[right]
        if abs(x2 - x1) < 1e-9: return 0, y1
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept

    # Check if the baseline crosses any significant valleys between the peak and boundaries.
    def _check_valley_crossing(self, left, right):
        if self.mode != 'strict': return False
        segment_r = self.signal[self.peak_center:right + 1]
        if len(segment_r) > 5:
            min_idx_r = np.argmin(segment_r)
            min_val_r = segment_r[min_idx_r]
            if self.signal[right] > min_val_r + 2.0 * self.noise_level:
                if min_idx_r < len(segment_r) - 3: return True
        segment_l = self.signal[left:self.peak_center + 1]
        if len(segment_l) > 5:
            min_idx_l = np.argmin(segment_l)
            min_val_l = segment_l[min_idx_l]
            if self.signal[left] > min_val_l + 2.0 * self.noise_level:
                if min_idx_l > 3: return True
        return False

    # Evaluate the tail regions for any significant deviation from the baseline.
    def _check_tail_status(self, left, right, baseline_slope, baseline_intercept):
        if self.mode == 'strict':
            return self._check_tail_strict(left, right, baseline_slope, baseline_intercept)
        else:
            return self._check_tail_simple(left, right, baseline_slope, baseline_intercept)

    # Perform a simple tail deviation check using a fixed width.
    def _check_tail_simple(self, left, right, m, c):
        check_width = 20
        tail_penalty = 0
        l_start = max(0, left - check_width)
        if left > l_start:
            y_real = self.signal[l_start:left]
            y_base = m * self.temperature[l_start:left] + c
            diff = y_real - y_base
            tail_penalty += np.sum(diff[diff > self.noise_level * 0.05])
        r_end = min(self.n_points, right + check_width)
        if r_end > right:
            y_real = self.signal[right:r_end]
            y_base = m * self.temperature[right:r_end] + c
            diff = y_real - y_base
            tail_penalty += np.sum(diff[diff > self.noise_level * 0.05])
        return tail_penalty

    # Perform a strict tail deviation check using signal gradients.
    def _check_tail_strict(self, left, right, m, c):
        check_width = 15
        tail_penalty = 0
        grads = np.gradient(self.signal)
        l_start = max(0, left - check_width)
        if left > l_start:
            diff = self.signal[l_start:left] - (m * self.temperature[l_start:left] + c)
            local_grads = grads[l_start:left]
            is_rising_prev = local_grads < -(np.max(np.abs(grads)) * 0.05)
            valid = (diff > self.noise_level) & (~is_rising_prev)
            if np.any(valid): tail_penalty += np.sum(diff[valid])
        r_end = min(self.n_points, right + check_width)
        if r_end > right:
            diff = self.signal[right:r_end] - (m * self.temperature[right:r_end] + c)
            local_grads = grads[right:r_end]
            is_rising_next = local_grads > (np.max(np.abs(grads)) * 0.05)
            valid = (diff > self.noise_level) & (~is_rising_next)
            if np.any(valid): tail_penalty += np.sum(diff[valid])
        return tail_penalty

    # Calculate the reinforcement learning reward for a given pair of boundaries.
    def _calculate_reward(self, left, right):
        score_log = []
        reward = 0

        if self._check_valley_crossing(left, right):
            return -10000

        slope, intercept = self._get_baseline_params(left, right)

        tail_resid = self._check_tail_status(left, right, slope, intercept)
        if tail_resid > 0:
            denom = self.y_span if self.mode == 'loose' else 1.0
            p_tail = 100 + (tail_resid / denom) * 100
            reward -= p_tail
            score_log.append(f"TailPenalty: -{p_tail:.1f}")
        else:
            reward += 50
            score_log.append("Tail: OK(+50)")

        idx = np.arange(left, right)
        if len(idx) > 0:
            base_vals = slope * self.temperature[idx] + intercept
            net_vals = self.signal[idx] - base_vals
            violation = net_vals[net_vals < -self.noise_level]

            if self.mode == 'strict':
                if len(violation) > 0:
                    p_viol = 500 + np.sum(np.abs(violation)) * 1000000
                    reward -= p_viol
                    score_log.append(f"Viol: -{p_viol:.0f}")

                area_weight = 0.2 if self.process_type == 'exo' else 0.1
                r_area = (np.sum(net_vals[net_vals > 0]) / self.y_span) * area_weight
                reward += r_area

                p_width = ((right - left) / self.n_points) * 200
                reward -= p_width

                dist_ratio = (abs(left - self.peak_center) + abs(right - self.peak_center)) / self.n_points
                reward -= dist_ratio * 50000
                score_log.append(f"Area: +{r_area:.1f} | WidthPen: -{p_width:.1f}")

            else:
                if len(violation) > 0:
                    p_viol = 100 + np.sum(np.abs(violation)) * 80
                    reward -= p_viol
                    score_log.append(f"Viol: -{p_viol:.1f}")

                r_area = (np.sum(net_vals[net_vals > 0]) / self.y_span) * 10
                reward += r_area

                r_width = ((right - left) / self.n_points) * 150
                reward += r_width
                score_log.append(f"Area: +{r_area:.1f} | WidthBonus: +{r_width:.1f}")

        scale = (self.temperature[-1] - self.temperature[0]) / self.y_span
        slope_diff = abs(slope - self.global_slope) * scale
        threshold = 0.5 if self.mode == 'loose' else 1.0
        if slope_diff > threshold:
            weight = 50 if self.mode == 'loose' else 20
            p_slope = (slope_diff ** 2) * weight
            reward -= p_slope
            score_log.append(f"SlopeDiff: -{p_slope:.1f}")

        grad = np.gradient(self.signal)
        g_weight = 1.0 if self.mode == 'strict' else 0.2
        grad_penalty = (abs(grad[left]) + abs(grad[right])) / np.max(np.abs(grad)) * g_weight
        reward -= grad_penalty
        score_log.append(f"GradPen: -{grad_penalty:.1f}")

        return reward

    # Execute the Q-learning algorithm to find the optimal peak boundaries.
    def boundary_optimization(self, max_iterations=100):
        print(f"\n=== Start RL optimization (Mode: {self.mode}) ===")
        print(f"Initial boundaries: Left={self.initial_left}, Right={self.initial_right}")

        q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        alpha, gamma = 0.3, 0.9
        epsilon = 0.6

        best_score = -float('inf')
        best_bounds = (self.initial_left, self.initial_right)
        left, right = self.initial_left, self.initial_right
        no_improv = 0

        for i in range(max_iterations):
            state = self._get_state(left, right)

            is_random = False
            if random.random() < epsilon:
                action = random.choice(self.actions)
                is_random = True
                act_str = f"RANDOM({action})"
            else:
                idx = np.argmax(q_table[state])
                action = self.actions[idx]
                act_str = f"GREEDY({action})"

            nl, nr = self._apply_action(left, right, action)
            nl = max(self.search_limit_left, nl)
            nr = min(self.search_limit_right, nr)
            if nr - nl < 20: nl, nr = left, right

            reward = self._calculate_reward(nl, nr)

            improv_mark = ""
            if reward > best_score:
                best_score = reward
                best_bounds = (nl, nr)
                no_improv = 0
                improv_mark = "★ NEW BEST"
            else:
                no_improv += 1

            print(
                f"Iter {i:03d} | Pos: [{left}->{nl}, {right}->{nr}] | {act_str:20s} | Rwd: {reward:8.1f} | Best: {best_score:8.1f} {improv_mark}")

            old_val = q_table[state][self.actions.index(action)]
            n_state = self._get_state(nl, nr)
            next_max = np.max(q_table[n_state])
            new_val = old_val + alpha * (reward + gamma * next_max - old_val)
            q_table[state][self.actions.index(action)] = new_val

            left, right = nl, nr
            if epsilon > 0.1: epsilon *= 0.96
            if no_improv > 25 and best_score > -5000:
                print(f"EARLY STOPPING: 50 steps no improvement.")
                break

        bl, br = best_bounds
        print(f"=== Optimization finished: Final boundaries [{bl}, {br}], Score {best_score:.2f} ===\n")

        slope, intercept = self._get_baseline_params(bl, br)
        baseline = slope * self.temperature + intercept
        idx = np.arange(bl, br)
        metric = 0
        if len(idx) > 0:
            metric = np.sqrt(np.mean((self.signal[idx] - baseline[idx]) ** 2))
        else:
            metric = 999
        return best_bounds, baseline, metric

    # Discretize the current boundary positions into a state representation.
    def _get_state(self, l, r):
        return (int(l / self.n_points * 20), int(r / self.n_points * 20))

    # Apply a given action to adjust the left and right boundaries.
    def _apply_action(self, l, r, action):
        if self.n_points < 800:
            step_small = 1
            step_large = 3
        else:
            step_small = 5
            step_large = 10

        nl, nr = l, r
        if action == "expand_left":
            nl = l - step_large
        elif action == "shrink_left":
            nl = l + step_small
        elif action == "expand_right":
            nr = r + step_large
        elif action == "shrink_right":
            nr = r - step_small
        elif action == "expand_both":
            nl, nr = l - step_large, r + step_large
        elif action == "shrink_both":
            nl, nr = l + step_small, r - step_small
        return nl, nr

# Perform baseline correction on Differential Scanning Calorimetry (DSC) data.
class BaselineCorrection():
    # Initialize the baseline correction object with temperature and heat capacity data.
    def __init__(self, Tm, Cp):
        self.Tm = Tm
        self.Cp = Cp

    # Identify the peak areas using an Exponential Gaussian Mixture model.
    def PeakArea(self, Tm, Cp, material_type=None):
        x = Cp
        mix = 0.5
        ParVariable = ExpGaussMix(a=1, z=mix)
        dom_condition1 = np.zeros(x.shape)
        for j in range(len(x)):
            ParVariable.SetOpt(True, True, True, True)
            ParVariable.Optimize(x[j], mix)
            ParChange = ParVariable.GetZ()
            print(f"The number of ParChange is {len(ParChange)}")
            for k in range(len(ParChange)):
                if ParChange[k] > mix + 0.06:
                    dom_condition1[j, k] = 1
        first_groups1 = process_dom_array(dom_condition1)
        Peak11 = []
        for i in range(len(first_groups1)):
            Peak11.append(np.mean(first_groups1[i]) if first_groups1[i] else 0)
        return np.array(Peak11), dom_condition1, first_groups1

    # Determine the optimal left and right boundaries for the identified peaks.
    def BePeak(self, Tm, Cp, fit_num=4, material_type=None, process_type='endo'):
        """
        BePeak with User-Specified Selection Rules:
        Exo: Left->Closer(Strict), Right->Farther(Loose)
        Endo: Left->Farther(Loose), Right->Closer(Strict)
        """
        x = Tm
        y = Cp
        Peak11, dom1, first_groups1 = self.PeakArea(x, y, material_type=material_type)
        BePeak11 = np.zeros(y.shape)
        AfPeak11 = np.zeros(y.shape)

        for i in range(y.shape[0]):
            if Peak11[i] > 0 and first_groups1[i]:
                peak_center = int(Peak11[i])
                print(f"Scan {i}: Performing bimodal optimization (Type={process_type})...")

                rl_loose = PeakBoundaryRL(x, y[i], peak_center, process_type=process_type, mode='loose')
                bounds_loose, _, _ = rl_loose.boundary_optimization()
                l_loose, r_loose = bounds_loose

                rl_strict = PeakBoundaryRL(x, y[i], peak_center, process_type=process_type, mode='strict')
                bounds_strict, _, _ = rl_strict.boundary_optimization()
                l_strict, r_strict = bounds_strict

                if process_type == 'exo':
                    final_left = l_strict
                    final_right = r_strict
                    print(f"  -> Exo rule applied: Both sides use closer boundaries (Strict)")
                else:
                    final_left = l_loose
                    final_right = r_loose
                    print(f"  -> Endo rule applied: Both sides use farther boundaries (Loose)")

                if final_left >= final_right:
                    print("  -> Warning: Boundaries crossed, falling back to Loose range")
                    final_left, final_right = l_loose, r_loose

                print(f"Scan {i}: Final boundaries: [{x[final_left]:.1f}, {x[final_right]:.1f}]")
                BePeak11[i, final_left] = 1
                AfPeak11[i, final_right] = 1

        return BePeak11, AfPeak11, dom1, first_groups1

    # Execute the full baseline correction process and extract the net signal.
    @staticmethod
    def BaseCorrect(Tm, Cp, fit_num=4, material_type=None, process_type='endo'):
        print("------Creating DSC Object------")
        P = BaselineCorrection(Tm, Cp)
        x = Tm
        y = Cp
        final_EMGMBas = np.copy(y)
        print(f"**Baseline correction (Type: {process_type})")

        BePeak11, AfPeak11, dom1, first_groups1 = P.BePeak(x, y, fit_num, material_type=material_type,
                                                           process_type=process_type)

        print("**Applying boundary refinement and calculating baseline")
        for i in range(y.shape[0]):
            if first_groups1[i]:
                be_indices = np.nonzero(BePeak11[i])[0]
                af_indices = np.nonzero(AfPeak11[i])[0]
                if len(be_indices) > 0 and len(af_indices) > 0:
                    mB = int(np.max(be_indices))
                    mA = int(np.min(af_indices))

                    shrink_amount = 3
                    mB = min(mB + shrink_amount, mA - 5)
                    mA = max(mA - shrink_amount, mB + 5)
                    mB = max(0, min(mB, len(x) - 1))
                    mA = max(0, min(mA, len(x) - 1))

                    BePeak11[i, :] = 0
                    AfPeak11[i, :] = 0
                    BePeak11[i, mB] = 1
                    AfPeak11[i, mA] = 1

                    left_temp, left_cp = x[mB], y[i, mB]
                    right_temp, right_cp = x[mA], y[i, mA]

                    if left_temp != right_temp:
                        slope = (right_cp - left_cp) / (right_temp - left_temp)
                        intercept = left_cp - slope * left_temp
                    else:
                        slope, intercept = 0, left_cp

                    for j in range(mB, mA + 1):
                        final_EMGMBas[i, j] = slope * x[j] + intercept

        ResiData = y - final_EMGMBas
        ResiData11 = np.zeros_like(y)
        for i in range(y.shape[0]):
            if first_groups1[i]:
                be = np.nonzero(BePeak11[i])[0]
                af = np.nonzero(AfPeak11[i])[0]
                if len(be) > 0 and len(af) > 0:
                    s, e = int(np.max(be)), int(np.min(af))
                    if s < e: ResiData11[i, s:e + 1] = ResiData[i, s:e + 1]

        def array_to_list_struct(source_array):
            p_list = []
            t_list = []
            for i in range(source_array.shape[0]):
                p_list.append([])
                p_list.append([])
                t_list.append([])
                indices = np.nonzero(source_array[i])[0]
                if len(indices) > 0:
                    s, e = np.min(indices), np.max(indices)
                    p_list[i] = source_array[i, s:e + 1].tolist()
                    t_list[i] = x[s:e + 1].tolist()
            return p_list, t_list

        Peak11, TemPeak11 = array_to_list_struct(ResiData11)
        return (ResiData11, ResiData, final_EMGMBas, dom1, Peak11, TemPeak11)