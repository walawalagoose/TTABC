from collections import defaultdict
import math

import numpy as np
import torch
from torch.utils.data import Sampler


def extract_dataset_labels(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        base_labels = extract_dataset_labels(dataset.dataset)
        return np.asarray(base_labels)[np.asarray(dataset.indices, dtype=np.int64)]

    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    elif hasattr(dataset, "label_list"):
        labels = dataset.label_list
    else:
        raise ValueError(
            f"label_shift is not supported for dataset type {type(dataset).__name__}: "
            "unable to locate labels/targets."
        )

    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().numpy().astype(np.int64)
    return np.asarray(labels, dtype=np.int64)


class LabelShiftBatchSampler(Sampler):
    def __init__(
        self,
        labels,
        batch_size,
        num_class_eff=None,
        num_class_eff_min=None,
        num_class_eff_max=None,
        num_batches=None,
        scope_size=-1,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.classes = np.array(sorted(np.unique(self.labels).tolist()), dtype=np.int64)
        self.num_class = len(self.classes)

        self.num_class_eff_min = num_class_eff_min
        self.num_class_eff_max = num_class_eff_max
        if num_class_eff_min is None and num_class_eff is not None:
            self.num_class_eff = int(num_class_eff)
        elif num_class_eff_min is None:
            self.num_class_eff = self.num_class
        else:
            self.num_class_eff = None

        self.num_batches = (
            int(num_batches)
            if num_batches is not None
            else max(1, math.ceil(len(self.labels) / max(1, self.batch_size)))
        )
        self.scope_size = int(scope_size) if scope_size is not None else -1

        self.label_dict = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_dict[int(label)].append(idx)

        self.indices = self._create_indices()

    def _sample_num_class_eff(self):
        if self.num_class_eff is None and self.num_class_eff_min is not None and self.num_class_eff_max is not None:
            return int(np.random.randint(self.num_class_eff_min, self.num_class_eff_max + 1))
        return int(self.num_class_eff)

    def _iter_scope_indices(self):
        total = len(self.labels)
        if self.scope_size is None or self.scope_size <= 0 or self.scope_size >= total:
            yield np.arange(total, dtype=np.int64)
            return
        for start in range(0, total, self.scope_size):
            end = min(start + self.scope_size, total)
            yield np.arange(start, end, dtype=np.int64)

    def _generate_batch(self, scope_indices):
        scope_labels = self.labels[scope_indices]
        scope_classes = np.array(sorted(np.unique(scope_labels).tolist()), dtype=np.int64)
        num_scope_class = len(scope_classes)
        num_class_eff = self._sample_num_class_eff()

        if 0 < num_class_eff < num_scope_class:
            selected_classes = np.random.choice(scope_classes, num_class_eff, replace=False)
        else:
            selected_classes = scope_classes

        all_indices = []
        for label in selected_classes:
            mask = scope_labels == int(label)
            all_indices.extend(scope_indices[mask].tolist())

        if len(all_indices) == 0:
            return []

        if self.batch_size == -1 or len(all_indices) <= self.batch_size:
            batch = list(all_indices)
        else:
            batch = np.random.choice(all_indices, self.batch_size, replace=False).tolist()

        np.random.shuffle(batch)
        return batch

    def _create_indices(self):
        stream = []
        for scope_indices in self._iter_scope_indices():
            if len(scope_indices) == 0:
                continue
            if self.batch_size == -1:
                num_batches_for_scope = 1
            else:
                num_batches_for_scope = max(1, math.ceil(len(scope_indices) / max(1, self.batch_size)))
            for _ in range(num_batches_for_scope):
                batch = self._generate_batch(scope_indices)
                if batch:
                    stream.extend(batch)
        return stream

    def __iter__(self):
        if self.batch_size == -1:
            if self.indices:
                yield self.indices
            return
        for start in range(0, len(self.indices), self.batch_size):
            batch = self.indices[start : start + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

    def __len__(self):
        if self.batch_size == -1:
            return 1 if self.indices else 0
        return len(self.indices) // self.batch_size


class LabelShiftOnlineBatchSampler(Sampler):
    def __init__(self, labels, gamma, slots, batch_size, scope_size=-1, seed=None):
        # A local generator makes Dirichlet draws reproducible under --seed.
        self.rng = np.random.default_rng(seed)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.scope_size = int(scope_size) if scope_size is not None else -1

        self.classes = np.array(sorted(np.unique(self.labels).tolist()), dtype=np.int64)
        self.num_class = len(self.classes)
        self.num_slots = int(slots) if slots is not None else self.num_class
        self.indices = self._create_indices()

    def _iter_scope_indices(self):
        total = len(self.labels)
        if self.scope_size is None or self.scope_size <= 0 or self.scope_size >= total:
            yield np.arange(total, dtype=np.int64)
            return
        for start in range(0, total, self.scope_size):
            end = min(start + self.scope_size, total)
            yield np.arange(start, end, dtype=np.int64)

    def _create_indices_for_scope(self, scope_indices):
        scope_labels = self.labels[scope_indices]
        scope_classes = np.array(sorted(np.unique(scope_labels).tolist()), dtype=np.int64)
        label_dict = defaultdict(list)
        for pos, label in zip(scope_indices, scope_labels):
            label_dict[int(label)].append(int(pos))

        if self.gamma == -1:
            final_indices = []
            shuffled_classes = np.random.permutation(scope_classes)
            for label in shuffled_classes:
                indices = np.array(label_dict[int(label)], dtype=np.int64)
                np.random.shuffle(indices)
                final_indices.extend(indices.tolist())
            return final_indices

        num_slots_local = max(1, min(self.num_slots, len(scope_indices)))
        label_distribution = self.rng.dirichlet([self.gamma] * num_slots_local, len(scope_classes))
        slot_indices = [[] for _ in range(num_slots_local)]

        for label_pos, label in enumerate(scope_classes):
            indices = np.array(label_dict[int(label)], dtype=np.int64)
            partition = label_distribution[label_pos, :]
            split_points = (np.cumsum(partition)[:-1] * len(indices)).astype(int)
            samples_per_slot = np.split(indices, split_points)
            for slot_id, ids in enumerate(samples_per_slot):
                slot_indices[slot_id].extend(ids.tolist())

        final_indices = []
        for slot in slot_indices:
            if not slot:
                continue
            permutation = np.random.permutation(len(slot))
            final_indices.extend([slot[i] for i in permutation])
        return final_indices

    def _create_indices(self):
        stream = []
        for scope_indices in self._iter_scope_indices():
            if len(scope_indices) == 0:
                continue
            scope_stream = self._create_indices_for_scope(scope_indices)
            stream.extend(scope_stream)
        return stream

    def __iter__(self):
        if self.batch_size == -1:
            if self.indices:
                yield self.indices
            return
        for start in range(0, len(self.indices), self.batch_size):
            batch = self.indices[start : start + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

    def __len__(self):
        if self.batch_size == -1:
            return 1 if self.indices else 0
        return len(self.indices) // self.batch_size
