import numpy as np


class ReservoirSample:

    def __init__(self):
        self.item = None
        self.weight = 0

    def update(self, item, weight):
        next_weight = self.weight + weight
        prob_flip = weight / next_weight
        self.weight = next_weight
        if np.random.random() < prob_flip:
            self.item = item


class BatchReservoirSample:

    def __init__(self):
        self.output_weight = 1
        self._dist_to_next_chosen = 0
        self._dist_to_next_block = 1

    def process_batch(self, data, max_items_to_sample):
        """Sample up to max_items_to_sample from a data array.

        Parameters
        ----------
        data: np.array
            Input data
        max_items_to_sample: int
            maximum number of items to sample

        Returns
        -------
        (sampled_indices, items_processed)
        sampled_indices : np.array of ints or None
            indices of items to sample
        items_processed : int
            number of items in the input data processed. Could be less than total number of items if we reached
            max_items_to_sample
        """
        # first calculate the items we need to processed. Modify the array if needed to make sure we will process the
        # full (resulting) array
        items_processed = self._dist_to_next_block + self.output_weight * (
                max_items_to_sample - (1 if self._dist_to_next_chosen >= 0 else 0))
        items_processed = min(data.shape[0], items_processed)
        data = data[:items_processed]

        if self.output_weight == 1:
            return np.arange(items_processed), items_processed

        first_index = None
        # insert item from first block that we are potentially in the middle of
        if self._dist_to_next_chosen >= 0:
            if self._dist_to_next_chosen >= items_processed:
                self._dist_to_next_chosen -= items_processed
                return None, items_processed
            else:
                first_index = np.array([self._dist_to_next_chosen], dtype=np.int)
        if self._dist_to_next_block > items_processed:
            self._dist_to_next_chosen -= items_processed
            self._dist_to_next_block -= items_processed
            return first_index, items_processed
        elif self._dist_to_next_block == items_processed:
            self._dist_to_next_chosen = np.random.randint(low=0, high=self.output_weight)
            self._dist_to_next_block = items_processed
            return first_index, items_processed

        # if we got here, there are more blocks to process.
        # insert items from blocks that are covered fully by this batch
        n_mid_blocks = (items_processed - self._dist_to_next_block) // self.output_weight
        if n_mid_blocks > 0:
            mid_indices = np.random.randint(low=0, high=self.output_weight, size=n_mid_blocks)
        else:
            mid_indices = []

        # insert item from the last block, not fully covered by this batch
        last_block_size = items_processed - (self._dist_to_next_block + n_mid_blocks * self.output_weight)
        self._dist_to_next_chosen = np.random.randint(low=0, high=self.output_weight) - last_block_size
        self._dist_to_next_block = items_processed - last_block_size
        if self._dist_to_next_chosen < 0:
            last_index = np.array([items_processed + self._dist_to_next_chosen])
        else:
            last_index = []

        # concatenate all the indices
        if first_index is None:
            first_index = []
        if len(first_index) + len(mid_indices) + len(last_index) == 0:
            return None, items_processed
        else:
            return np.concatenate([first_index, mid_indices, last_index], axis=0).astype(np.int), items_processed

    def double_output_weight(self):
        if self._dist_to_next_chosen >= 0:
            self._dist_to_next_block += self.output_weight
            if np.random.randint(low=0, high=2) == 1:
                self._dist_to_next_chosen = self._dist_to_next_block - np.random.randint(low=1,
                                                                                         high=self.output_weight + 1)
        self.output_weight *= 2

    def update(self, weight):
        assert weight <= self.output_weight and self.output_weight > 1, \
            'currently we only support adding items whose weight is upper bounded by output weight'
        ret_count = 0
        if self._dist_to_next_chosen >= weight:
            self._dist_to_next_chosen -= weight
            self._dist_to_next_block -= weight
        elif self._dist_to_next_block > weight:
            ret_count = self._dist_to_next_chosen >= 0
            self._dist_to_next_block -= weight
            self._dist_to_next_chosen -= weight
        else:
            ret_count = self._dist_to_next_chosen >= 0
            new_weight = weight - self._dist_to_next_block
            self._dist_to_next_block = self.output_weight
            self._dist_to_next_chosen = np.random.randint(low=0, high=self.output_weight)
            ret_count += self.update(new_weight)
        return ret_count

    def items_processed_in_block(self):
        return self.output_weight * (self._dist_to_next_chosen < 0)


class BiasedParams:
    def __init__(self):
        self.min_size = 8
        self.min_ratio_to_compact = 0.33
        self.always_leave_out = 2
        # with large coef_for_barrier, we are typically choosing less items to be compacted,
        # thereby paying more for mid-value quantiles but gaining for the heavier tails
        self.coef_for_barrier = 8
        # same logic for coef_for_barrier, but a different way of distributing the weight. Also, the effect
        # is inverse. a small value gives more weight to heavy tails
        self.power_for_barrier = 0.8


class KLLR:
    """KLL algorithm with hyper-parameters to turn on/off various strategies.

    Parameters
    ----------
    sketch_size : int
        The overall number of items the sketch is allowed to store. The sketch will actually store this + 1 more item
        for a reservoir sampler

    sampling_strategy : str ('lazy' or 'strict')
        Strategy for using the sampling. Either only when we have to (lazy) or in a strict way for the bottom layers

    compact_strategy : str ('potential', 'first')
        strategy by which we choose the level to compact when we are out of space

    biased : bool, default False
        If true we run the biased version trying to preserve top quantiles more aggresively than bottom / medium ones.
        If false, we run the version that gives the same attention to all quantiles

    error_spreading : string ('never', 'always', 'adaptive')
        Strategy for error spreading.
        When biased==True, this if forced to be 'never'

    correlated_compact : bool, default True
        Determines whether the compactions are forced to be correlated to have opposite shifts between pairs or
        uniformly random shifts

    sweep : bool, default False
        Indicator to whether we perform a sweep compaction

    sweep_correlated_compact : bool, default True
        Only valid when sweep==True
        If false, each pair will be compacted via different randomness. Else, all pairs will be compacted the same way
        (smaller or larger is kept) throughout a sweep

    """

    def __init__(self, sketch_size=200, sampling_strategy='lazy', compact_strategy='potential', biased=False,
                 error_spreading='adaptive', correlated_compact=True, sweep=False, sweep_correlated_compact=True,
                 bias_param=None):
        assert sketch_size >= 4, 'minimum supported sketch size is 4'

        # hyperparameters
        self._sampling_strategy = sampling_strategy
        self._compact_strategy = compact_strategy
        self._biased = biased
        self._error_spreading = error_spreading if not biased else 'never'
        self._correlated_compact = correlated_compact
        self._sweep = sweep
        self._sweep_correlated_compact = sweep_correlated_compact
        self._bias_param = BiasedParams() if bias_param is None else bias_param

        # min capacity for the smallest compactor
        self._initial_min_compactor_size = 4 + (self._error_spreading == 'always')

        self.data = np.empty(shape=(sketch_size, ))
        # information about the levels. Starting point in the array and minimum capacity
        self._starts = [0]
        self._ends = [0]
        self._min_capacity = [self._top_min_capacity]
        self._sweep_barrier = [-np.inf]
        self._sweep_untouched = [0]

        # information required for sampling
        if self._sampling_strategy == 'lazy':
            self.sampler = ReservoirSample()
        elif self._sampling_strategy == 'strict':
            self.sampler = BatchReservoirSample()
        else:
            raise ValueError(f'unknown value for sampling strategy {sampling_strategy}')

        # for debugging mostly keep track of the number of items processed
        self.n = 0

        # needed for trick with correlated randomness
        self.rand_bit_info = {}

    @property
    def _top_min_capacity(self):
        """
        Returns the capacity of the top compactor

        The compactor at level h-1 has 2/3 the capacity of that of level h. We would like to make sure that the
        capacities are integers and products of 2, so we set them according to the rule
            capacity[h-1] = capacity[h] // 3 * 2
        The inverse is
            capacity[h] = capacity[h-1] * 3 // 2
            capacity[h] += capacity[h] % 2
        We start from self._initial_min_compactor_size, the minimum capacity and increase by maintaining the
        cummulative sum. We keep going until exceeding the memory limit, and return the last capacity just before that
        happens
        """
        cumsum = self._initial_min_compactor_size
        k = self._initial_min_compactor_size
        while cumsum <= self.data.shape[0]:
            k = k * 3 // 2
            k += k % 2
            cumsum += k
        high = k
        low = (k // 3) * 2
        med = (high + low) // 2
        while low < high:
            med = (high + low) // 2
            k = med
            mem_at_med = 0
            while k >= self._initial_min_compactor_size:
                mem_at_med += k
                k = (k // 3) * 2

            k = med + 1
            mem_at_medp1 = 0
            while k >= self._initial_min_compactor_size:
                mem_at_medp1 += k
                k = (k // 3) * 2

            if mem_at_med <= self.data.shape[0] < mem_at_medp1:
                break
            elif mem_at_med <= self.data.shape[0]:
                low = med + 1
            else:
                high = med

        return med

    def update(self, data):
        """Updates the sketch with an array of data. Quantiles are maintained per column

        Parameters
        ----------
        data : numpy 1d array
            new batch of data to feed into the sketch. Rows are data points
        """
        data = data.reshape((-1,))

        # keep track of number of items seen (might only be needed for debugging)
        self.n += data.shape[0]

        start_ind = 0
        while start_ind < data.shape[0]:
            # compute capacity. If 0, compress to increase it
            if self._l0_capacity() <= 0:
                self._compress()

            # if the standard compression didn't do the trick, collect all layers with capacity 2
            if self._l0_capacity() <= 0:
                assert self._sampling_strategy == 'lazy'
                self._collect_cap2_layers()
            # append data to level 0.
            start_ind = self._update_l0_has_space(start_ind, data)

    def _update_l0_has_space(self, start_ind, data):
        # update the sketch when we know that level 0 has space in it. The update data is data[start_ind:]
        cap = self._l0_capacity()
        assert cap > 0
        if self._sampling_strategy == 'lazy':
            next_start_ind = min(start_ind + cap, data.shape[0])
            self._l0_append(data[start_ind: next_start_ind])
            return next_start_ind
        else:
            data = data[start_ind:]
            sampled_indices, items_processed = self.sampler.process_batch(data, cap)
            if sampled_indices is not None:
                self._l0_append(data[sampled_indices])
            return start_ind + items_processed

    def _collect_cap2_layers(self):

        # find the layer index of the first layer with min_capacity of self._initial_min_compactor_size.
        # This layer cannot be collected

        first_protected = -1
        for layer in range(len(self._starts)):
            if self._min_capacity[layer] >= self._initial_min_compactor_size:
                first_protected = layer
                break
        assert first_protected > 0, f'{self._min_capacity}, {self._starts}'

        # iterate over the layers. For those that have an item in them, delete it and add it to a
        # reservour sampling sketch
        n_freed = 0
        for layer in range(first_protected):
            n_items = self._ends[layer] - self._starts[layer]
            if n_items > 0:
                # TODO(zkarnin) - this makes sense only if self._initial_min_compactor_size=4. Otherwise, the condition
                #  in the assertion might actually happen, and we need to figure out what to do in such a case.
                assert n_items <= 2
                for i in range(n_items):
                    item = self.data[self._starts[layer] + i].copy()
                    self.sampler.update(item, weight=2 ** layer)
                self._starts[layer] = -1 if layer > 0 else 0
                self._ends[layer] = -1 if layer > 0 else 0
                n_freed += n_items
            # this is a hack of sorts. To optimize for min error we should stop once we freed 2 items. However,
            # for speed we should free more. The logic below is a compromise that gives some speed at very little cost
            # to the error
            if n_freed > 4 and layer > first_protected - 4:
                break

        # find the first non-empty layer
        nelayer = 0
        for layer in range(first_protected):
            n_items = self._ends[layer] - self._starts[layer]
            if n_items > 0:
                nelayer = layer
                break
        if nelayer == 0:
            nelayer = first_protected
        assert self._starts[nelayer] > 0

        # if the weight of the sampler is large enough, add a number to this layer
        added = False
        if self.sampler.weight >= 2**nelayer:
            added = True
            item = self.sampler.item.copy()
            self.sampler.weight -= 2**nelayer
            self._starts[nelayer] -= 1
            self.data[self._starts[nelayer]] = item
        if added:
            # sort the layer
            self._sort_mid_level(nelayer)

        # fix the starts and ends of the empty layers
        for layer in range(1, nelayer):
            self._starts[layer] = self._starts[nelayer]
            self._ends[layer] = self._starts[nelayer]

    def merge(self, other):
        raise NotImplementedError

    def _start(self, level):
        if level < len(self._starts):
            return self._starts[level]
        return self.data.shape[0]

    def _l0_capacity(self):
        return self._start(1) - self._ends[0]

    def _l0_append(self, data):
        assert self._ends[0] >= 0, f'{self._ends}'
        self.data[self._ends[0]: self._ends[0] + data.shape[0]] = data
        self._ends[0] += data.shape[0]

    def _new_rand_shift(self, key):
        if not self._correlated_compact:
            self.rand_bit_info[key] = [np.random.randint(0, 2), 0]
        else:
            self._next_rand_bit(key, force_no_sweep=True)

    def _next_rand_bit(self, key, force_no_sweep=False):
        if self._sweep and not force_no_sweep:
            if key not in self.rand_bit_info:
                self.rand_bit_info[key] = [np.random.randint(0, 2), 0]
            return self.rand_bit_info[key][0] if self._sweep_correlated_compact else np.random.randint(0,2)
        if self._correlated_compact:
            if key not in self.rand_bit_info:
                self.rand_bit_info[key] = [np.random.randint(0,2), 0]
            if self.rand_bit_info[key][1] == 0:
                self.rand_bit_info[key][1] = 1
                ret = self.rand_bit_info[key][0]
                self.rand_bit_info[key][0] = 1 - ret
                return ret
            else:
                self.rand_bit_info[key][1] = 1
                ret = self.rand_bit_info[key][0]
                self.rand_bit_info[key][0] = np.random.randint(0, 2)
                return ret
        else:
            return np.random.randint(0, 2)

    def _single_compact_get_subset(self, lstart, lend, force_full_compaction):
        # according to the strategy figure out the start and end indices of the items we aim to get rid of
        rbit = np.random.randint(low=0, high=2)
        if self._error_spreading == 'always':
            if force_full_compaction and ((lend - lstart) % 2 == 0):
                new_start = lstart
                new_end = lend
            else:
                new_start = lstart + rbit
                new_end = lend - ((lend - lstart + rbit) % 2)
        elif self._error_spreading == 'never':
            new_start = lstart
            new_end = lend - ((lend - lstart) % 2)
        elif self._error_spreading == 'adaptive':
            new_start = lstart + rbit * ((lend - lstart) % 2)
            new_end = lend - (1 - rbit) * ((lend - lstart) % 2)
        else:
            raise ValueError(f'unknown value for error_spreading: {self._error_spreading}')

        if self._biased:
            # TODO(zkarnin) - this is hacky... put some HPs here, consider the options
            length = lend - lstart
            if length >= self._bias_param.min_size:
                min_ratio_to_compact = max(2, int(length * self._bias_param.min_ratio_to_compact))
                smallest_end = lstart + min_ratio_to_compact
                smallest_end += (smallest_end - lstart) % 2
                largest_end = lend - self._bias_param.always_leave_out
                largest_end -= (largest_end - lstart) % 2

                cand_end = np.array(list(range(smallest_end, largest_end + 1, 2)))

                probs = (cand_end - lstart) / (largest_end - smallest_end)
                probs = probs ** self._bias_param.power_for_barrier
                probs = np.exp(-self._bias_param.coef_for_barrier * probs)
                probs /= np.sum(probs)
                new_end = cand_end[np.argmax(np.random.multinomial(1, probs))]
                new_start = lstart

        return new_start, new_end

    def _single_compact_get_subset_sweep(self, level, lstart, lend):
        barrier = self._sweep_barrier[level]
        if np.isneginf(barrier):
            # we need to start a new sweep. Get a new random shift
            self._new_rand_shift(level)
            # when running in biased mode, don't want the sweep to run through the entire buffer. This is the point
            # where we need to set an endpoint for the sweep. We use the same logic as in the non-sweep mode to
            # figure out how many items should be left untouched by this next sweep
            if self._biased:
                new_start, new_end = self._single_compact_get_subset(lstart, lend, force_full_compaction=False)
                num_untuched = lend - new_end
                self._sweep_untouched[level] = num_untuched
            else:
                self._sweep_untouched[level] = 0

        # find the leftmost (smallest) item x with x >= barrier
        idx = np.searchsorted(self.data[lstart:lend], barrier) + lstart
        if idx >= lend - 1 - self._sweep_untouched[level]:
            # we finished the sweep - we cannot compact this pair. Start a new sweep
            # safety measure - make sure there are at least two items in the buffer, otherwise we'll enter an infinite
            # loop
            assert lend - lstart >= 2
            self._sweep_barrier[level] = -np.inf
            return self._single_compact_get_subset_sweep(level, lstart, lend)
        else:
            self._sweep_barrier[level] = self.data[idx + 1]
            return idx, idx + 2

    def _handle_kept_items(self, level, lstart, lend, force_full_compaction):
        if self._sweep and not force_full_compaction:
            new_start, new_end = self._single_compact_get_subset_sweep(level, lstart, lend)
        else:
            new_start, new_end = self._single_compact_get_subset(lstart, lend, force_full_compaction)

        # move items around so that items we're getting rid of are at the right and those we keep are on the left
        if new_end < lend:
            right_kept = self.data[new_end:lend].copy()
            shift = lend - new_end
            self.data[new_start + shift:lend] = self.data[new_start:new_end]
            self.data[new_start:new_start+shift] = right_kept
            new_start += shift
            new_end += shift

        return new_start

    def _compact_single_level(self, level, force_full_compaction=False):
        lstart = self._starts[level]
        lend = self._start(level + 1)
        assert level == 0 or lend == self._ends[level], f'level={level}, {lend} != {self._ends[level]}'

        # sometimes we will choose not to compact all the items. This function will move the kept items to the left,
        # make sure both partitions, kept and non-kept, are sorted, and return the start index of the items to be
        # compacted. In particular, it makes sure that lend - new_start is even
        new_start = self._handle_kept_items(level, lstart, lend, force_full_compaction)

        # The items we output are either the odd indexed or even indexed.
        # This is done by having a pointer (j) starting either in the last location or second to last at random
        # i: index pointing to the last location before the next layer
        # j: last index of layer - offset (random 0 or 1)

        i = lend - 1
        j = i - self._next_rand_bit(level, force_no_sweep=force_full_compaction)

        while j >= new_start:
            self.data[i] = self.data[j]
            i -= 1
            j -= 2

        # update information about this level
        self._ends[level] = new_start

        # update info for next level
        if level + 1 < len(self._starts):
            self._starts[level + 1] = i + 1
            return -1
        else:
            return i + 1

    def _update_internals_new_level(self, first_idx):
        # reduce the capacity of all existing layers
        for i in range(len(self._min_capacity)):
            self._min_capacity[i] = max(2 + (self._error_spreading == 'always'), self._min_capacity[i] // 3 * 2)

        # add another entry to buffer
        if self._sampling_strategy == 'lazy' or self._min_capacity[0] >= self._initial_min_compactor_size:
            self._starts.append(first_idx)
            self._ends.append(self.data.shape[0])
            self._min_capacity.append(self._top_min_capacity)
            self._sweep_barrier.append(-np.inf)
            self._sweep_untouched.append(0)
        elif self._sampling_strategy == 'strict':
            # we added a new layer - this means that sampling needs to output items of twice the weight.
            # First, we double the output weight of the sampler
            self.sampler.double_output_weight()

            # we need to empty out level 0 completely, as it contains items whose weight is too small
            if self._ends[0] > self._starts[0]:
                self._compact_single_level(0, force_full_compaction=True)
                if self._ends[0] - self._starts[0] > 0:
                    assert self._starts[1] >= 2, 'just compacted - there should be at least 2 vacant spots in the' \
                                                 f' sketch, but there is just {self._starts[1] - self._ends[0]}'
                    assert self._ends[0] - self._starts[0] == 1 and self._starts[0] == 0
                    l1_item_count = self.sampler.update(self.sampler.output_weight // 2)
                    if l1_item_count > 0:
                        self._starts[1] -= l1_item_count
                        self.data[self._starts[1]:self._starts[1]+l1_item_count] = l1_item_count

            # now, let's move level 1 to start at zero since it is going to be the new level 0
            l1_len = self._ends[1] - self._starts[1]
            if self._starts[1] > 0:
                self.data[:l1_len] = self.data[self._starts[1]:self._ends[1]]
                self._starts[1] = 0
                self._ends[1] = l1_len

            # we're now ready to shift the internal values one to the left
            for i in range(0, len(self._starts) - 1):
                self._starts[i] = self._starts[i + 1]
                self._ends[i] = self._ends[i + 1]
                self._min_capacity[i] = self._min_capacity[i + 1]
                self._sweep_barrier[i] = self._sweep_barrier[i + 1]
                self._sweep_untouched[i] = self._sweep_untouched[i + 1]
                try:
                    self.rand_bit_info[i] = self.rand_bit_info[i + 1]
                except KeyError:
                    pass

            # insert the internal values for the top layer
            self._starts[-1] = first_idx
            self._ends[-1] = self.data.shape[0]
            self._min_capacity[-1] = self._top_min_capacity
            self._sweep_barrier[-1] = -np.inf
            self._sweep_untouched[-1] = 0

        else:
            raise ValueError(f'unknown sampling_strategy {self._sampling_strategy}')

    def _sort_mid_level(self, level):
        start = self._starts[level]
        end = self._ends[level]
        # TODO(zkarnin) - replace this with merge. To do this we need to pass the number of new items just inserted
        self.data[start:end] = np.sort(self.data[start:end], axis=0)

    def _sortl0(self):
        start = self._starts[0]
        end = self._start(1)
        if end > start:
            self.data[start:end] = np.sort(self.data[start:end], axis=0)

    def _best_compact_layer(self):
        cur_max_gain = 1e-20
        best_level = -1

        if self._compact_strategy == 'potential':
            coeff = int((3 / 2) ** len(self._starts))
            for level in range(len(self._starts)):
                size = self._ends[level] - self._starts[level]
                size -= size % 2

                cur_gain = size * coeff * (size >= self._min_capacity[level])

                if cur_gain > cur_max_gain:
                    cur_max_gain = cur_gain
                    best_level = level

                coeff = (coeff * 2 + 2) // 3
        elif self._compact_strategy == 'first':
            for level in range(len(self._starts)):
                size = self._ends[level] - self._starts[level]
                size -= size % 2
                if size >= self._min_capacity[level]:
                    best_level = level
                    break
        else:
            raise ValueError(f'unknown value for compact_strategy {self._compact_strategy}')

        assert self._ends[best_level] - self._starts[best_level] >= 2, \
            f'size={self._ends[best_level] - self._starts[best_level]}, min_cap={self._min_capacity[best_level]},' \
            f' cur_max_gain={cur_max_gain}'

        return best_level

    def _compress(self):
        # compresses the sketch by choosing a layer to compact. The chosen layer is that maximizing a potential
        # function. This function is either one that chooses the first layer with more items than min capacity or one
        # corresponding (intuitively at least, since the computation is not precise) to the inverse variance on
        # the added rank error due to the compact operation, multiplied with a discount factor biasing us towards the
        # lower layers.
        # The gain function for level h is defined as
        #   gain(h) = (2/3)**h * num_items(h) * 1[num_items(h) >= min_capacity]
        # here, num_items(h) is the number of items in the h'th layer.
        # 1[num_items(h,s) >= min_capacity] is an 0/1 indicator to whether the number of items is larger than the
        # capacity. This term ensures we do not compact a level that doesn't have sufficiently many items.

        # make sure the items are ordered. They are kept ordered except for level 0
        self._sortl0()

        # find the best layer to compact
        best_level = self._best_compact_layer()

        # it could be the case we opened too many levels, and none of them can be compacted. In this case we do not
        # compress at all
        if best_level < 0:
            return

        # compact the level. With some settings this might not entirely eliminate the level but only some items in it.
        # The guarantee is that if there are still item left, the last index of the level is just before the first item
        # of the next non-empty level, and items that should remain in this level are in consecutive memory slots.
        # If the level we just compacted was the top level and items were outputted, this means we just opened a new
        # level. In this case, this function does not update the meta-data of start positions, etc., for this new level.
        # In order for this to be done the function returns the starting index of the new level if it exists and -1
        # otherwise.
        new_level_first_idx = self._compact_single_level(best_level)

        # by compacting a layer that is potentially not the bottom one we may have some layer i whose end is not
        # adjacent to the beginning of layer i+1. This function makes sure to push all items in the array, thereby
        # keeping the invariant that only layer 0 might end before layer 1.
        self._ensure_data_adjacency(best_level + 1, new_level_first_idx)

        if new_level_first_idx >= 0:
            # If we had to open a new level we update parameters
            self._update_internals_new_level(new_level_first_idx)
        else:
            # We didn't open a new level, so we need to make sure the level beyond is sorted
            # Note: this can be done with inplace merge (see https://www.geeksforgeeks.org/in-place-merge-sort/)
            #       because the levels above 0 are always sorted. Unfortunately numpy does not have an inplace sort
            #       function
            self._sort_mid_level(best_level + 1)

    def _ensure_data_adjacency(self, top_level, new_level_first_idx):
        assert top_level == len(self._starts) or self._starts[top_level] != -1, \
            f'top_level={top_level}, starts={self._starts}, self._starts[top_level]={self._starts[top_level]}'

        for level in range(top_level - 1, 0, -1):
            next_start = self._starts[level + 1] if level + 1 < len(self._starts) else new_level_first_idx
            if self._ends[level] != next_start:
                new_start = next_start - (self._ends[level] - self._starts[level])
                new_end = next_start
                assert (self._starts[level] != -1 and self._ends[level] != -1) or \
                       (self._ends[level] == -1 and self._ends[level] == -1), \
                    f'{self._starts[level]}, {self._ends[level]}'
                if self._ends[level] - self._starts[level] > 0:
                    # cpy = self.data[self._starts[level]:self._ends[level]].copy()
                    self.data[new_start: new_end] = self.data[self._starts[level]:self._ends[level]]
                    # assert np.linalg.norm(self.data[new_start: new_end] - cpy) == 0
                self._starts[level] = new_start
                self._ends[level] = new_end
                assert level > 0 and new_end >= 0 and new_start >= 0

        assert all([s >= 0 for s in self._starts])
        assert all([s >= 0 for s in self._ends])
        assert sum([abs(end - start) for start, end in zip(self._starts[2:], self._ends[1:-1])]) == 0,\
                f'{self._starts}\n{self._ends}'

    def rank(self, x):
        """

        Parameters
        ----------
        x: float or np.array
            A number or row whose rank we wish to compute.

        Returns
        -------
        The rank of x. If x is a row we return a row containing the ranks of its elements w.r.t to the different columns
        of the input data we observed

        """
        if self._sampling_strategy == 'strict':
            cur_weight = self.sampler.output_weight
        else:
            cur_weight = 1
        rank = 0
        n = 0
        for level in range(len(self._starts)):
            cur_rank = (x >= self.data[self._starts[level]:self._ends[level]]).sum()
            rank += cur_weight * cur_rank
            n += (self._ends[level] - self._starts[level]) * cur_weight
            cur_weight *= 2

        return rank / n

    def item_seen(self):
        n = 0
        cur_weight = 1 if self._sampling_strategy == 'lazy' else self.sampler.output_weight

        for level in range(len(self._starts)):
            cur_len = (self._ends[level] - self._starts[level])
            n += cur_len * cur_weight
            cur_weight *= 2
        if self._sampling_strategy == 'lazy':
            n += self.sampler.weight
        elif self._sampling_strategy == 'strict':
            n += self.sampler.items_processed_in_block()
        return n

    def get_value(self, rank):
        n = self.item_seen()
        relevant_data = self.data[list(range(self._ends[0])) + list(range(self._starts[1], self._ends[-1]))]

        cur_weight = 1 if self._sampling_strategy == 'lazy' else self.sampler.output_weight
        weights = []
        for level in range(len(self._starts)):
            cur_len = (self._ends[level] - self._starts[level])
            weights.extend([cur_weight] * cur_len)
            cur_weight *= 2

        abs_rank = int(n * rank)
        val = None
        idx = np.argsort(relevant_data)
        cur_rank = 0
        for id in idx:
            cur_rank += weights[id]
            if cur_rank >= abs_rank:
                val = relevant_data[id]
                break
        if val is None:
            val = relevant_data[idx[-1]]
        return val


def main():
    mem = 1000

    batch = 500
    n_batch = 500
    col_num = 1

    import time

    tstart = time.time()

    kll = KLLR(mem, sampling_strategy='lazy', compact_strategy='potential', error_spreading='adaptive',
               correlated_compact=True, biased=True, sweep=False, sweep_correlated_compact=True)

    n = batch * n_batch * col_num
    full_data = np.arange(n * col_num).reshape((n, col_num))
    full_data = np.flip(full_data)
    for i in range(col_num):
        col0 = full_data[:, i]
        np.random.shuffle(col0)
        full_data[:, i] = col0
    # full_data += np.random.randint(low=-50, high=50, size=(n, 1))*100

    for i in range(n_batch):
        data = full_data[i * batch: (i + 1) * batch]
        kll.update(data)
        # kll.update(data, int(n * 0.95))
        # assert kll._sampling_strategy != 'lazy' or \
        #        kll.item_seen() == (i + 1) * batch, f'{kll.item_seen()} != {(i + 1) * batch}'

    full_data = np.sort(full_data, axis=0)

    max_err = 0
    for r in range(0, n, n // 100):
        ground_truth = full_data[r]
        err = abs(r/n - kll.rank(ground_truth))
        # err = abs((kll.get_value(r / n) - ground_truth) / n)
        max_err = np.maximum(max_err, err)
    print('max error = ', max_err)

    for q in [0.995, 0.999, 0.9997]:
        ground_truth = full_data[int(n * q)]
        err = abs((kll.get_value(q) - ground_truth) / n)
        # err = abs(q - kll.rank(ground_truth))
        relative = err / (1 - q)
        print(f'relative error for querying p{q * 100} is {relative}')

    runtime = time.time() - tstart
    print('runtime = ', runtime)


if __name__ == '__main__':
    main()