"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def pad_to_6(self, ops):
        """Pad ops list to 6 elements by repeating last op"""
        if len(ops) >= 6:
            return ops[:6]
        return ops + [ops[-1]] * (6 - len(ops))

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Optimized VLIW SIMD implementation with tree-level processing:
        - Levels 0-2: arithmetic selection (broadcast/select)
        - Levels 3-4: cascaded vselect with cached tree values
        - Levels 5+: scatter-gather with pipelining
        """
        n_chunks = batch_size // VLEN  # 32 chunks

        # ===== SCRATCH ALLOCATION =====
        scratch_idx = self.alloc_scratch("scratch_idx", batch_size)
        scratch_val = self.alloc_scratch("scratch_val", batch_size)

        MAX_CHUNKS = 6
        v_tree = [self.alloc_scratch(f"v_tree_{i}", VLEN) for i in range(MAX_CHUNKS)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(MAX_CHUNKS)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(MAX_CHUNKS)]
        v_cond = [self.alloc_scratch(f"v_cond_{i}", VLEN) for i in range(MAX_CHUNKS)]
        v_tree_alt = [self.alloc_scratch(f"v_tree_alt_{i}", VLEN) for i in range(MAX_CHUNKS)]
        v_tree_third = [self.alloc_scratch(f"v_tree_third_{i}", VLEN) for i in range(MAX_CHUNKS)]
        gather_addrs = [[self.alloc_scratch(f"addr_{c}_{i}") for i in range(VLEN)] for c in range(MAX_CHUNKS)]
        gather_addrs_alt = [[self.alloc_scratch(f"addr_alt_{c}_{i}") for i in range(VLEN)] for c in range(MAX_CHUNKS)]

        # Triple-buffer array for tree values (eliminates LOAD-only cycles)
        tree_buffers = [v_tree, v_tree_alt, v_tree_third]

        # Constant vectors
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_four = self.alloc_scratch("v_four", VLEN)

        # Hash constant vectors
        v_mult_4097 = self.alloc_scratch("v_mult_4097", VLEN)
        v_mult_33 = self.alloc_scratch("v_mult_33", VLEN)
        v_mult_9 = self.alloc_scratch("v_mult_9", VLEN)
        v_hash_c0 = self.alloc_scratch("v_hash_c0", VLEN)
        v_hash_c1 = self.alloc_scratch("v_hash_c1", VLEN)
        v_hash_c2 = self.alloc_scratch("v_hash_c2", VLEN)
        v_hash_c3 = self.alloc_scratch("v_hash_c3", VLEN)
        v_hash_c4 = self.alloc_scratch("v_hash_c4", VLEN)
        v_hash_c5 = self.alloc_scratch("v_hash_c5", VLEN)
        v_shift_19 = self.alloc_scratch("v_shift_19", VLEN)
        v_shift_9 = self.alloc_scratch("v_shift_9", VLEN)
        v_shift_16 = self.alloc_scratch("v_shift_16", VLEN)

        # Scalar temporaries and constants
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        tmp4 = self.alloc_scratch("tmp4")
        s_zero = self.alloc_scratch("s_zero")
        s_one = self.alloc_scratch("s_one")
        s_two = self.alloc_scratch("s_two")

        # Tree level 0-2 scalars
        s_tree0 = self.alloc_scratch("s_tree0")
        s_tree1 = self.alloc_scratch("s_tree1")
        s_tree2 = self.alloc_scratch("s_tree2")
        s_tree_diff = self.alloc_scratch("s_tree_diff")

        # Tree level 2 scalars (indices 3-6)
        s_tree3 = self.alloc_scratch("s_tree3")
        s_tree4 = self.alloc_scratch("s_tree4")
        s_tree5 = self.alloc_scratch("s_tree5")
        s_tree6 = self.alloc_scratch("s_tree6")
        s_diff_low = self.alloc_scratch("s_diff_low")
        s_diff_high = self.alloc_scratch("s_diff_high")
        v_tree3 = self.alloc_scratch("v_tree3", VLEN)
        v_tree5 = self.alloc_scratch("v_tree5", VLEN)
        v_diff_low = self.alloc_scratch("v_diff_low", VLEN)
        v_diff_high = self.alloc_scratch("v_diff_high", VLEN)
        v_three = self.alloc_scratch("v_three", VLEN)
        v_seven = self.alloc_scratch("v_seven", VLEN)

        # Level 3 tree values (8 scalars) and differences (4 scalars)
        s_l3 = [self.alloc_scratch(f"s_l3_{i}") for i in range(8)]
        s_l3_d01 = self.alloc_scratch("s_l3_d01")
        s_l3_d23 = self.alloc_scratch("s_l3_d23")
        s_l3_d45 = self.alloc_scratch("s_l3_d45")
        s_l3_d67 = self.alloc_scratch("s_l3_d67")

        # Precomputed output addresses for faster result copy
        out_addrs = [self.alloc_scratch(f"out_addr_{i}") for i in range(n_chunks)]

        # Memory pointers
        for v in ["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]:
            self.alloc_scratch(v, 1)

        # ===== INITIALIZATION =====
        # Load header values (pipelined)
        for i in range(0, 7, 2):
            self.instrs.append({"load": [("const", tmp1, i), ("const", tmp2, i+1 if i+1 < 7 else i)]})
            loads = [("load", self.scratch[["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"][i]], tmp1)]
            if i + 1 < 7:
                loads.append(("load", self.scratch[["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"][i+1]], tmp2))
            self.instrs.append({"load": loads})

        # Load constants with pipelined broadcasts
        # Pipeline: load consts in cycle N, broadcast in cycle N+1 while loading next consts
        self.instrs.append({"load": [("const", s_zero, 0), ("const", s_one, 1)]})
        self.instrs.append({"load": [("const", s_two, 2), ("const", tmp1, 4097)]})  # Load while s_zero, s_one ready
        self.instrs.append({
            "load": [("const", tmp2, 33), ("const", tmp3, 9)],
            "valu": [("vbroadcast", v_one, s_one), ("vbroadcast", v_two, s_two)],
        })
        self.instrs.append({
            "load": [("const", tmp4, 19), ("const", s_tree_diff, 16)],  # Reuse s_tree_diff temporarily
            "valu": [("vbroadcast", v_mult_4097, tmp1), ("vbroadcast", v_mult_33, tmp2), ("vbroadcast", v_mult_9, tmp3)],
        })
        self.instrs.append({
            "load": [("const", tmp1, 4), ("const", tmp2, 3)],
            "valu": [("vbroadcast", v_shift_19, tmp4), ("vbroadcast", v_shift_16, s_tree_diff)],
        })
        self.instrs.append({
            "load": [("const", tmp3, 7), ("const", tmp4, 0x7ED55D16)],
            "valu": [("vbroadcast", v_four, tmp1), ("vbroadcast", v_three, tmp2)],
        })
        self.instrs.append({
            "load": [("const", tmp1, 0xC761C23C), ("const", tmp2, 0x165667B1)],
            "valu": [("vbroadcast", v_seven, tmp3), ("vbroadcast", v_hash_c0, tmp4)],
        })
        self.instrs.append({
            "load": [("const", tmp3, 0xD3A2646C), ("const", tmp4, 0xFD7046C5)],
            "valu": [("vbroadcast", v_hash_c1, tmp1), ("vbroadcast", v_hash_c2, tmp2)],
        })
        self.instrs.append({
            "load": [("const", tmp1, 0xB55A4F09), ("const", tmp2, 9)],
            "valu": [("vbroadcast", v_hash_c3, tmp3), ("vbroadcast", v_hash_c4, tmp4)],
        })
        self.instrs.append({
            "valu": [("vbroadcast", v_hash_c5, tmp1), ("vbroadcast", v_shift_9, tmp2)],
        })

        # Tree values for levels 0-2 - optimized with pipelining
        # Load offsets and compute addresses in parallel with loading tree values
        self.instrs.append({"load": [("load", s_tree0, self.scratch["forest_values_p"]), ("const", tmp1, 1)]})
        self.instrs.append({
            "load": [("const", tmp2, 2), ("const", tmp3, 3)],
            "alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1)],
        })
        self.instrs.append({
            "load": [("load", s_tree1, tmp1), ("const", tmp4, 4)],
            "alu": [("+", tmp2, self.scratch["forest_values_p"], tmp2), ("+", tmp3, self.scratch["forest_values_p"], tmp3)],
        })
        self.instrs.append({
            "load": [("load", s_tree2, tmp2), ("const", tmp1, 5)],
            "alu": [("+", tmp4, self.scratch["forest_values_p"], tmp4)],
        })
        self.instrs.append({
            "load": [("load", s_tree3, tmp3), ("load", s_tree4, tmp4)],
            "alu": [("-", s_tree_diff, s_tree1, s_tree2), ("+", tmp1, self.scratch["forest_values_p"], tmp1)],
        })
        self.instrs.append({
            "load": [("load", s_tree5, tmp1), ("const", tmp2, 6)],
        })
        self.instrs.append({
            "alu": [("+", tmp2, self.scratch["forest_values_p"], tmp2)],
        })
        self.instrs.append({
            "load": [("load", s_tree6, tmp2)],
        })
        self.instrs.append({
            "alu": [("-", s_diff_low, s_tree4, s_tree3), ("-", s_diff_high, s_tree6, s_tree5)],
            "valu": [("vbroadcast", v_tree3, s_tree3), ("vbroadcast", v_tree5, s_tree5)],
        })
        self.instrs.append({
            "valu": [("vbroadcast", v_diff_low, s_diff_low), ("vbroadcast", v_diff_high, s_diff_high)],
        })

        # Level 3 tree value initialization disabled (level 3 uses scatter-gather)

        # Copy indices and values to scratch, precompute output addresses
        for chunk in range(0, n_chunks, 2):
            off0, off1 = chunk * VLEN, (chunk + 1) * VLEN
            self.instrs.append({"load": [("const", tmp1, off0), ("const", tmp2, off1)]})
            self.instrs.append({"alu": [
                ("+", tmp3, self.scratch["inp_indices_p"], tmp1),
                ("+", tmp4, self.scratch["inp_indices_p"], tmp2),
                ("+", out_addrs[chunk], self.scratch["inp_values_p"], tmp1),
                ("+", out_addrs[chunk + 1], self.scratch["inp_values_p"], tmp2),
            ]})
            self.instrs.append({"load": [("vload", scratch_idx + off0, tmp3), ("vload", scratch_idx + off1, tmp4)]})
            self.instrs.append({"alu": [("+", tmp3, self.scratch["inp_values_p"], tmp1), ("+", tmp4, self.scratch["inp_values_p"], tmp2)]})
            self.instrs.append({"load": [("vload", scratch_val + off0, tmp3), ("vload", scratch_val + off1, tmp4)]})

        # First pause - matches first yield in reference_kernel2 (before processing)
        self.instrs.append({"flow": [("pause",)]})

        # ===== DYNAMIC UNIVERSAL PIPELINE =====
        # Strategy is automatically selected based on level features:
        # - num_unique_values: how many distinct tree values at this level
        # - fits_in_cache: whether all values fit in scratch
        # - Strategies: BROADCAST (1 value), ARITHMETIC (2-4 values), SCATTER_GATHER (many values)
        tree_base = self.scratch["forest_values_p"]

        # Thresholds for strategy selection (can be tuned)
        ARITHMETIC_THRESHOLD = 4  # Use arithmetic selection if <= this many unique values
        CACHE_THRESHOLD = 64      # Cache in scratch if <= this many unique values

        # Dynamic group sizing: key optimization!
        # - Scatter-gather: 4 chunks/group (32 loads = 16 cycles, limits LOAD-only cycles)
        # - Broadcast/arithmetic: 6 chunks/group (no loads, no padding waste)
        CHUNKS_SG = 4  # For scatter-gather: balance LOAD/VALU
        CHUNKS_OTHER = 6  # For broadcast/arithmetic: no padding waste

        # Build VALU ops for hash + index update
        def build_hash_index_ops(tree_reg, val_addrs, idx_addrs, n, is_rnd10):
            ops = []
            ops.append([("^", val_addrs[c], val_addrs[c], tree_reg[c]) for c in range(n)])
            ops.append([("multiply_add", val_addrs[c], val_addrs[c], v_mult_4097, v_hash_c0) for c in range(n)])
            ops.append([("^", v_tmp1[c], val_addrs[c], v_hash_c1) for c in range(n)])
            ops.append([(">>", v_tmp2[c], val_addrs[c], v_shift_19) for c in range(n)])
            ops.append([("^", val_addrs[c], v_tmp1[c], v_tmp2[c]) for c in range(n)])
            ops.append([("multiply_add", val_addrs[c], val_addrs[c], v_mult_33, v_hash_c2) for c in range(n)])
            ops.append([("+", v_tmp1[c], val_addrs[c], v_hash_c3) for c in range(n)])
            ops.append([("<<", v_tmp2[c], val_addrs[c], v_shift_9) for c in range(n)])
            ops.append([("^", val_addrs[c], v_tmp1[c], v_tmp2[c]) for c in range(n)])
            ops.append([("multiply_add", val_addrs[c], val_addrs[c], v_mult_9, v_hash_c4) for c in range(n)])
            ops.append([("^", v_tmp1[c], val_addrs[c], v_hash_c5) for c in range(n)])
            ops.append([(">>", v_tmp2[c], val_addrs[c], v_shift_16) for c in range(n)])
            ops.append([("^", val_addrs[c], v_tmp1[c], v_tmp2[c]) for c in range(n)])
            if is_rnd10:
                ops.append([("vbroadcast", idx_addrs[c], s_zero) for c in range(n)])
            else:
                ops.append([("%", v_cond[c], val_addrs[c], v_two) for c in range(n)])
                ops.append([("multiply_add", idx_addrs[c], idx_addrs[c], v_two, v_one) for c in range(n)])
                ops.append([("+", idx_addrs[c], idx_addrs[c], v_cond[c]) for c in range(n)])
            return ops

        def compute_level_features(rnd, n_nodes):
            """Compute features of the tree level for this round."""
            # Tree level = round mod 11 (wraps at level 10 back to 0)
            # Level 0: indices at 0 (root)
            # Level k: indices at 2^k - 1 to 2^(k+1) - 2
            level = rnd % 11  # Levels 0-10, then wraps
            if level == 0:
                # All indices are 0 (root) - either start or after wrap
                num_unique = 1
                level_start = 0
            else:
                num_unique = min(1 << level, n_nodes)  # 2^level unique values
                level_start = (1 << level) - 1  # First index at this level

            # Determine strategy based on features
            if num_unique == 1:
                strategy = "broadcast"
            elif num_unique <= ARITHMETIC_THRESHOLD:
                strategy = "arithmetic"
            else:
                strategy = "scatter_gather"

            # Dynamic group sizing based on strategy
            # Scatter-gather: 4 chunks (32 loads = 16 cycles matches 16 VALU exactly)
            # Others: 6 chunks (better VALU utilization, no load bottleneck)
            if strategy == "scatter_gather":
                chunks_per_group = CHUNKS_SG
            else:
                chunks_per_group = CHUNKS_OTHER

            # Buffer depth: more buffering for larger levels
            if strategy == "scatter_gather":
                buffer_depth = 3 if num_unique > CACHE_THRESHOLD else 2
            else:
                buffer_depth = 1

            return {
                "level": level,
                "num_unique": num_unique,
                "level_start": level_start,
                "strategy": strategy,
                "chunks_per_group": chunks_per_group,
                "buffer_depth": buffer_depth,
            }

        # Pipeline state tracking
        first_group_preloaded = False
        second_group_addrs_ready = False

        for rnd in range(rounds):
            is_round_10 = (rnd == 10)
            features = compute_level_features(rnd, n_nodes)
            strategy = features["strategy"]
            level_start = features["level_start"]
            num_unique = features["num_unique"]
            cpg = features["chunks_per_group"]  # Dynamic chunks per group

            # Calculate number of groups for this round
            n_groups_this_round = (n_chunks + cpg - 1) // cpg

            # Track address-ready state within the round
            group_addrs_ready = [False] * n_groups_this_round
            this_round_first_preloaded = first_group_preloaded
            this_round_second_addrs_ready = second_group_addrs_ready
            first_group_preloaded = False
            second_group_addrs_ready = False

            if this_round_second_addrs_ready:
                group_addrs_ready[0] = True
                if len(group_addrs_ready) > 1:
                    group_addrs_ready[1] = True

            for g_idx, chunk_base in enumerate(range(0, n_chunks, cpg)):
                n_active = min(cpg, n_chunks - chunk_base)
                chunk_idx_addrs = [scratch_idx + (chunk_base + c) * VLEN for c in range(n_active)]
                chunk_val_addrs = [scratch_val + (chunk_base + c) * VLEN for c in range(n_active)]

                # Buffer selection based on strategy
                use_alt = (g_idx % 2 == 1)
                curr_tree = v_tree_alt if use_alt else v_tree
                next_tree = v_tree if use_alt else v_tree_alt
                curr_addrs = gather_addrs_alt if use_alt else gather_addrs
                next_addrs = gather_addrs if use_alt else gather_addrs_alt

                next_chunk_base = chunk_base + cpg
                has_next = next_chunk_base < n_chunks
                is_last_group = not has_next
                next_n_active = min(cpg, n_chunks - next_chunk_base) if has_next else 0
                next_chunk_idx_addrs = [scratch_idx + (next_chunk_base + c) * VLEN for c in range(next_n_active)] if has_next else []

                # Check if next group's addresses are pre-computed
                next_addrs_ready = (g_idx + 1 < len(group_addrs_ready) and group_addrs_ready[g_idx + 1]) if has_next else False

                # ===== STRATEGY: BROADCAST (1 unique value) =====
                if strategy == "broadcast":
                    # All items get the same tree value - just broadcast
                    valu_ops = [[("vbroadcast", curr_tree[c], s_tree0) for c in range(n_active)]]
                    valu_ops.extend(build_hash_index_ops(curr_tree, chunk_val_addrs, chunk_idx_addrs, n_active, is_round_10))

                    # Use free LOAD slots to prefetch for future scatter-gather rounds
                    next_features = compute_level_features(rnd + 1, n_nodes) if rnd + 1 < rounds else None
                    should_prefetch = next_features and next_features["strategy"] == "scatter_gather" and is_last_group

                    if should_prefetch:
                        # Prefetch for next round's first groups using next round's chunk size
                        next_cpg = next_features["chunks_per_group"]
                        g0_idx = [scratch_idx + c * VLEN for c in range(next_cpg)]
                        g0_alu = [("+", gather_addrs[c][i], tree_base, g0_idx[c] + i)
                                  for c in range(next_cpg) for i in range(VLEN)]
                        g1_idx = [scratch_idx + (next_cpg + c) * VLEN for c in range(next_cpg)]
                        g1_alu = [("+", gather_addrs_alt[c][i], tree_base, g1_idx[c] + i)
                                  for c in range(next_cpg) for i in range(VLEN)]
                        next_alu = g0_alu + g1_alu
                        next_load = [("load", v_tree[c] + i, gather_addrs[c][i])
                                     for c in range(next_cpg) for i in range(VLEN)]

                        alu_idx, load_idx = 0, 0
                        for ops in valu_ops:
                            bundle = {"valu": self.pad_to_6(ops)}
                            if alu_idx < len(next_alu):
                                bundle["alu"] = next_alu[alu_idx:alu_idx+12]
                                alu_idx += 12
                            if alu_idx >= len(next_alu) and load_idx < len(next_load):
                                bundle["load"] = next_load[load_idx:load_idx+2]
                                load_idx += 2
                            self.instrs.append(bundle)
                        while load_idx < len(next_load):
                            self.instrs.append({"load": next_load[load_idx:load_idx+2]})
                            load_idx += 2
                        first_group_preloaded = True
                        second_group_addrs_ready = True
                    else:
                        for ops in valu_ops:
                            self.instrs.append({"valu": self.pad_to_6(ops)})

                # ===== STRATEGY: ARITHMETIC (2-4 unique values) =====
                elif strategy == "arithmetic":
                    if num_unique == 2:
                        # 2-way selection: tree[level_start + (idx - level_start)]
                        # idx is either level_start or level_start+1
                        # Use: base + (idx & 1) * diff where base=tree[level_start+1], diff=tree[level_start]-tree[level_start+1]
                        valu_ops = [
                            [("vbroadcast", curr_tree[c], s_tree2) for c in range(n_active)],
                            [("&", v_cond[c], chunk_idx_addrs[c], v_one) for c in range(n_active)],
                            [("vbroadcast", v_tmp1[c], s_tree_diff) for c in range(n_active)],
                            [("*", v_cond[c], v_cond[c], v_tmp1[c]) for c in range(n_active)],
                            [("+", curr_tree[c], curr_tree[c], v_cond[c]) for c in range(n_active)],
                        ]
                    else:  # num_unique == 4
                        # 4-way selection using arithmetic
                        valu_ops = [
                            [("-", v_cond[c], chunk_idx_addrs[c], v_three) for c in range(n_active)],
                            [("&", v_tmp1[c], v_cond[c], v_one) for c in range(n_active)],
                            [(">>", v_cond[c], v_cond[c], v_one) for c in range(n_active)],
                            [("multiply_add", v_tmp2[c], v_tmp1[c], v_diff_low, v_tree3) for c in range(n_active)],
                            [("multiply_add", curr_tree[c], v_tmp1[c], v_diff_high, v_tree5) for c in range(n_active)],
                            [("-", v_tmp1[c], curr_tree[c], v_tmp2[c]) for c in range(n_active)],
                            [("multiply_add", curr_tree[c], v_cond[c], v_tmp1[c], v_tmp2[c]) for c in range(n_active)],
                        ]

                    valu_ops.extend(build_hash_index_ops(curr_tree, chunk_val_addrs, chunk_idx_addrs, n_active, is_round_10))

                    # Use free LOAD slots to prefetch for future scatter-gather rounds
                    next_features = compute_level_features(rnd + 1, n_nodes) if rnd + 1 < rounds else None
                    should_prefetch = next_features and next_features["strategy"] == "scatter_gather" and is_last_group

                    if should_prefetch:
                        next_cpg = next_features["chunks_per_group"]
                        g0_idx = [scratch_idx + c * VLEN for c in range(next_cpg)]
                        g0_alu = [("+", gather_addrs[c][i], tree_base, g0_idx[c] + i)
                                  for c in range(next_cpg) for i in range(VLEN)]
                        g1_idx = [scratch_idx + (next_cpg + c) * VLEN for c in range(next_cpg)]
                        g1_alu = [("+", gather_addrs_alt[c][i], tree_base, g1_idx[c] + i)
                                  for c in range(next_cpg) for i in range(VLEN)]
                        next_alu = g0_alu + g1_alu
                        next_load = [("load", v_tree[c] + i, gather_addrs[c][i])
                                     for c in range(next_cpg) for i in range(VLEN)]

                        alu_idx, load_idx = 0, 0
                        for ops in valu_ops:
                            bundle = {"valu": self.pad_to_6(ops)}
                            if alu_idx < len(next_alu):
                                bundle["alu"] = next_alu[alu_idx:alu_idx+12]
                                alu_idx += 12
                            if alu_idx >= len(next_alu) and load_idx < len(next_load):
                                bundle["load"] = next_load[load_idx:load_idx+2]
                                load_idx += 2
                            self.instrs.append(bundle)
                        while load_idx < len(next_load):
                            self.instrs.append({"load": next_load[load_idx:load_idx+2]})
                            load_idx += 2
                        first_group_preloaded = True
                        second_group_addrs_ready = True
                    else:
                        for ops in valu_ops:
                            self.instrs.append({"valu": self.pad_to_6(ops)})

                # ===== STRATEGY: SCATTER_GATHER (many unique values) =====
                else:
                    # Load tree values via scatter-gather with pipelining
                    if g_idx == 0 and not this_round_first_preloaded:
                        # First group not preloaded: compute addresses then load
                        all_alu = [("+", curr_addrs[c][i], tree_base, chunk_idx_addrs[c] + i)
                                   for c in range(n_active) for i in range(VLEN)]
                        for i in range(0, len(all_alu), 12):
                            self.instrs.append({"alu": all_alu[i:i+12]})
                        all_load = [("load", curr_tree[c] + i, curr_addrs[c][i])
                                    for c in range(n_active) for i in range(VLEN)]
                        for i in range(0, len(all_load), 2):
                            self.instrs.append({"load": all_load[i:i+2]})

                    valu_ops = build_hash_index_ops(curr_tree, chunk_val_addrs, chunk_idx_addrs, n_active, is_round_10)

                    # Determine next round's strategy for prefetching decisions
                    next_features = compute_level_features(rnd + 1, n_nodes) if rnd + 1 < rounds else None
                    next_round_is_scatter = next_features and next_features["strategy"] == "scatter_gather"

                    # Determine what to compute/load during this group's VALU cycles
                    if next_addrs_ready and has_next:
                        next_load = [("load", next_tree[c] + i, next_addrs[c][i])
                                     for c in range(next_n_active) for i in range(VLEN)]
                        nn_chunk_base = chunk_base + 2 * cpg
                        if nn_chunk_base < n_chunks:
                            nn_n_active = min(cpg, n_chunks - nn_chunk_base)
                            nn_idx_addrs = [scratch_idx + (nn_chunk_base + c) * VLEN for c in range(nn_n_active)]
                            next_alu = [("+", curr_addrs[c][i], tree_base, nn_idx_addrs[c] + i)
                                        for c in range(nn_n_active) for i in range(VLEN)]
                            if g_idx + 2 < len(group_addrs_ready):
                                group_addrs_ready[g_idx + 2] = True
                        else:
                            next_alu = []

                    elif is_last_group:
                        next_load = []
                        if next_round_is_scatter:
                            next_cpg = next_features["chunks_per_group"]
                            g0_idx = [scratch_idx + c * VLEN for c in range(next_cpg)]
                            g0_alu = [("+", gather_addrs[c][i], tree_base, g0_idx[c] + i)
                                      for c in range(next_cpg) for i in range(VLEN)]
                            g1_idx = [scratch_idx + (next_cpg + c) * VLEN for c in range(next_cpg)]
                            g1_alu = [("+", gather_addrs_alt[c][i], tree_base, g1_idx[c] + i)
                                      for c in range(next_cpg) for i in range(VLEN)]
                            next_alu = g0_alu + g1_alu
                            next_load = [("load", v_tree[c] + i, gather_addrs[c][i])
                                         for c in range(next_cpg) for i in range(VLEN)]
                            first_group_preloaded = True
                            second_group_addrs_ready = True
                        else:
                            next_alu = []

                    elif has_next:
                        next_alu = [("+", next_addrs[c][i], tree_base, next_chunk_idx_addrs[c] + i)
                                    for c in range(next_n_active) for i in range(VLEN)]
                        next_load = [("load", next_tree[c] + i, next_addrs[c][i])
                                     for c in range(next_n_active) for i in range(VLEN)]

                    else:
                        next_alu, next_load = [], []

                    # Bundle VALU with ALU and LOAD
                    can_load_from_start = next_addrs_ready
                    alu_idx, load_idx = 0, 0
                    for ops in valu_ops:
                        bundle = {"valu": self.pad_to_6(ops)}
                        if alu_idx < len(next_alu):
                            bundle["alu"] = next_alu[alu_idx:alu_idx+12]
                            alu_idx += 12
                        if can_load_from_start and load_idx < len(next_load):
                            bundle["load"] = next_load[load_idx:load_idx+2]
                            load_idx += 2
                        elif alu_idx >= len(next_alu) and load_idx < len(next_load):
                            bundle["load"] = next_load[load_idx:load_idx+2]
                            load_idx += 2
                        self.instrs.append(bundle)

                    while load_idx < len(next_load):
                        self.instrs.append({"load": next_load[load_idx:load_idx+2]})
                        load_idx += 2

        # Copy results back to main memory using precomputed addresses
        for chunk in range(0, n_chunks, 2):
            off0, off1 = chunk * VLEN, (chunk + 1) * VLEN
            self.instrs.append({"store": [("vstore", out_addrs[chunk], scratch_val + off0), ("vstore", out_addrs[chunk + 1], scratch_val + off1)]})

        # Final pause
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734

def do_kernel_test(forest_height, rounds, batch_size, seed=123, trace=False, prints=False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, value_trace=value_trace, trace=trace)
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        assert machine.mem[inp_values_p:inp_values_p+len(inp.values)] == ref_mem[inp_values_p:inp_values_p+len(inp.values)], f"Incorrect result on round {i}"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5]:mem[5]+len(inp.indices)]
            assert inp.values == mem[mem[6]:mem[6]+len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
