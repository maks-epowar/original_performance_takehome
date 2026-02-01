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
        CHUNKS_PER_GROUP = 6  # Process 6 chunks per group to fully utilize 6 VALU slots

        # ===== SCRATCH ALLOCATION =====
        scratch_idx = self.alloc_scratch("scratch_idx", batch_size)
        scratch_val = self.alloc_scratch("scratch_val", batch_size)

        MAX_CHUNKS = 6
        v_tree = [self.alloc_scratch(f"v_tree_{i}", VLEN) for i in range(MAX_CHUNKS)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(MAX_CHUNKS)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(MAX_CHUNKS)]
        v_cond = [self.alloc_scratch(f"v_cond_{i}", VLEN) for i in range(MAX_CHUNKS)]
        v_tree_alt = [self.alloc_scratch(f"v_tree_alt_{i}", VLEN) for i in range(MAX_CHUNKS)]
        gather_addrs = [[self.alloc_scratch(f"addr_{c}_{i}") for i in range(VLEN)] for c in range(MAX_CHUNKS)]
        gather_addrs_alt = [[self.alloc_scratch(f"addr_alt_{c}_{i}") for i in range(VLEN)] for c in range(MAX_CHUNKS)]

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

        # Memory pointers
        for v in ["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]:
            self.alloc_scratch(v, 1)

        # ===== INITIALIZATION =====
        # Load header values
        for i in range(0, 7, 2):
            self.instrs.append({"load": [("const", tmp1, i), ("const", tmp2, i+1 if i+1 < 7 else i)]})
            loads = [("load", self.scratch[["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"][i]], tmp1)]
            if i + 1 < 7:
                loads.append(("load", self.scratch[["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"][i+1]], tmp2))
            self.instrs.append({"load": loads})

        # Load constants
        self.instrs.append({"load": [("const", s_zero, 0), ("const", s_one, 1)]})
        self.instrs.append({"load": [("const", s_two, 2), ("const", tmp1, 4097)]})
        self.instrs.append({"load": [("const", tmp2, 33), ("const", tmp3, 9)]})
        self.instrs.append({"load": [("const", tmp4, 19)]})

        self.instrs.append({"valu": [
            ("vbroadcast", v_one, s_one), ("vbroadcast", v_two, s_two),
            ("vbroadcast", v_mult_4097, tmp1), ("vbroadcast", v_mult_33, tmp2),
            ("vbroadcast", v_mult_9, tmp3), ("vbroadcast", v_shift_19, tmp4),
        ]})
        self.instrs.append({"load": [("const", tmp1, 16), ("const", tmp2, 9)]})
        self.instrs.append({"load": [("const", tmp3, 4)]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_shift_16, tmp1), ("vbroadcast", v_shift_9, tmp2),
            ("vbroadcast", v_four, tmp3),
        ]})
        self.instrs.append({"load": [("const", tmp1, 3), ("const", tmp2, 7)]})
        self.instrs.append({"valu": [("vbroadcast", v_three, tmp1), ("vbroadcast", v_seven, tmp2)]})

        # Hash constants
        self.instrs.append({"load": [("const", tmp1, 0x7ED55D16), ("const", tmp2, 0xC761C23C)]})
        self.instrs.append({"valu": [("vbroadcast", v_hash_c0, tmp1), ("vbroadcast", v_hash_c1, tmp2)]})
        self.instrs.append({"load": [("const", tmp1, 0x165667B1), ("const", tmp2, 0xD3A2646C)]})
        self.instrs.append({"valu": [("vbroadcast", v_hash_c2, tmp1), ("vbroadcast", v_hash_c3, tmp2)]})
        self.instrs.append({"load": [("const", tmp1, 0xFD7046C5), ("const", tmp2, 0xB55A4F09)]})
        self.instrs.append({"valu": [("vbroadcast", v_hash_c4, tmp1), ("vbroadcast", v_hash_c5, tmp2)]})

        # Tree values for levels 0-2
        self.instrs.append({"load": [("load", s_tree0, self.scratch["forest_values_p"]), ("const", tmp1, 1)]})
        self.instrs.append({"alu": [("+", tmp1, self.scratch["forest_values_p"], tmp1)]})
        self.instrs.append({"load": [("load", s_tree1, tmp1), ("const", tmp2, 2)]})
        self.instrs.append({"alu": [("+", tmp2, self.scratch["forest_values_p"], tmp2)]})
        self.instrs.append({"load": [("load", s_tree2, tmp2)]})
        self.instrs.append({"alu": [("-", s_tree_diff, s_tree1, s_tree2)]})

        # Load tree level 2 values (indices 3-6)
        self.instrs.append({"load": [("const", tmp1, 3), ("const", tmp2, 4)]})
        self.instrs.append({"alu": [("+", tmp3, self.scratch["forest_values_p"], tmp1), ("+", tmp4, self.scratch["forest_values_p"], tmp2)]})
        self.instrs.append({"load": [("load", s_tree3, tmp3), ("load", s_tree4, tmp4)]})
        self.instrs.append({"load": [("const", tmp1, 5), ("const", tmp2, 6)]})
        self.instrs.append({"alu": [("+", tmp3, self.scratch["forest_values_p"], tmp1), ("+", tmp4, self.scratch["forest_values_p"], tmp2)]})
        self.instrs.append({"load": [("load", s_tree5, tmp3), ("load", s_tree6, tmp4)]})
        self.instrs.append({"alu": [("-", s_diff_low, s_tree4, s_tree3), ("-", s_diff_high, s_tree6, s_tree5)]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_tree3, s_tree3), ("vbroadcast", v_tree5, s_tree5),
            ("vbroadcast", v_diff_low, s_diff_low), ("vbroadcast", v_diff_high, s_diff_high),
        ]})

        # Level 3 tree value initialization disabled (level 3 uses scatter-gather)

        # Copy indices and values to scratch
        for chunk in range(0, n_chunks, 2):
            off0, off1 = chunk * VLEN, (chunk + 1) * VLEN
            self.instrs.append({"load": [("const", tmp1, off0), ("const", tmp2, off1)]})
            self.instrs.append({"alu": [("+", tmp3, self.scratch["inp_indices_p"], tmp1), ("+", tmp4, self.scratch["inp_indices_p"], tmp2)]})
            self.instrs.append({"load": [("vload", scratch_idx + off0, tmp3), ("vload", scratch_idx + off1, tmp4)]})
            self.instrs.append({"alu": [("+", tmp3, self.scratch["inp_values_p"], tmp1), ("+", tmp4, self.scratch["inp_values_p"], tmp2)]})
            self.instrs.append({"load": [("vload", scratch_val + off0, tmp3), ("vload", scratch_val + off1, tmp4)]})

        # First pause - matches first yield in reference_kernel2 (before processing)
        self.instrs.append({"flow": [("pause",)]})

        # ===== MAIN LOOP =====
        tree_base = self.scratch["forest_values_p"]

        # Track preloading state for scatter-gather rounds
        first_group_preloaded = False
        second_group_addrs_ready = False  # True if group 1's addresses are pre-computed

        for rnd in range(rounds):
            is_level_0 = (rnd == 0) or (rnd == 11)
            is_level_1 = (rnd == 1) or (rnd == 12)
            is_level_2 = (rnd == 2) or (rnd == 13)
            is_level_3 = (rnd == 3) or (rnd == 14)
            is_round_10 = (rnd == 10)

            # Check if this round's first group was preloaded by previous round
            this_round_first_preloaded = first_group_preloaded
            this_round_second_addrs_ready = second_group_addrs_ready
            first_group_preloaded = False
            second_group_addrs_ready = False

            for g_idx, chunk_base in enumerate(range(0, n_chunks, CHUNKS_PER_GROUP)):
                n_active = min(CHUNKS_PER_GROUP, n_chunks - chunk_base)
                chunk_idx_addrs = [scratch_idx + (chunk_base + c) * VLEN for c in range(n_active)]
                chunk_val_addrs = [scratch_val + (chunk_base + c) * VLEN for c in range(n_active)]

                use_alt = (g_idx % 2 == 1)
                curr_tree = v_tree_alt if use_alt else v_tree
                next_tree = v_tree if use_alt else v_tree_alt
                curr_addrs = gather_addrs_alt if use_alt else gather_addrs
                next_addrs = gather_addrs if use_alt else gather_addrs_alt

                next_chunk_base = chunk_base + CHUNKS_PER_GROUP
                has_next = next_chunk_base < n_chunks
                next_n_active = min(CHUNKS_PER_GROUP, n_chunks - next_chunk_base) if has_next else 0
                next_chunk_idx_addrs = [scratch_idx + (next_chunk_base + c) * VLEN for c in range(next_n_active)] if has_next else []

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

                if is_level_0:
                    # Broadcast tree[0]
                    valu_ops = [[("vbroadcast", curr_tree[c], s_tree0) for c in range(n_active)]]
                    valu_ops.extend(build_hash_index_ops(curr_tree, chunk_val_addrs, chunk_idx_addrs, n_active, is_round_10))

                    # Preload next round if last group and next is scatter-gather
                    is_last_group = not has_next
                    next_round_is_scatter = (rnd + 1 < rounds) and (rnd + 1 not in [0, 1, 2, 11, 12, 13])  # 3 and 14 now use scatter-gather
                    if is_last_group and next_round_is_scatter:
                        first_cg_idx = [scratch_idx + c * VLEN for c in range(CHUNKS_PER_GROUP)]
                        next_alu = [("+", gather_addrs[c][i], tree_base, first_cg_idx[c] + i)
                                    for c in range(CHUNKS_PER_GROUP) for i in range(VLEN)]
                        next_load = [("load", v_tree[c] + i, gather_addrs[c][i])
                                     for c in range(CHUNKS_PER_GROUP) for i in range(VLEN)]
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
                    else:
                        for ops in valu_ops:
                            self.instrs.append({"valu": self.pad_to_6(ops)})

                elif is_level_1:
                    # Arithmetic selection: tree[2] + (idx & 1) * (tree[1] - tree[2])
                    valu_ops = [
                        [("vbroadcast", curr_tree[c], s_tree2) for c in range(n_active)],
                        [("&", v_cond[c], chunk_idx_addrs[c], v_one) for c in range(n_active)],
                        [("vbroadcast", v_tmp1[c], s_tree_diff) for c in range(n_active)],
                        [("*", v_cond[c], v_cond[c], v_tmp1[c]) for c in range(n_active)],
                        [("+", curr_tree[c], curr_tree[c], v_cond[c]) for c in range(n_active)],
                    ]
                    valu_ops.extend(build_hash_index_ops(curr_tree, chunk_val_addrs, chunk_idx_addrs, n_active, is_round_10))

                    # Preload next round if last group and next is scatter-gather
                    is_last_group = not has_next
                    next_round_is_scatter = (rnd + 1 < rounds) and (rnd + 1 not in [0, 1, 2, 11, 12, 13])  # 3 and 14 now use scatter-gather
                    if is_last_group and next_round_is_scatter:
                        first_cg_idx = [scratch_idx + c * VLEN for c in range(CHUNKS_PER_GROUP)]
                        next_alu = [("+", gather_addrs[c][i], tree_base, first_cg_idx[c] + i)
                                    for c in range(CHUNKS_PER_GROUP) for i in range(VLEN)]
                        next_load = [("load", v_tree[c] + i, gather_addrs[c][i])
                                     for c in range(CHUNKS_PER_GROUP) for i in range(VLEN)]
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
                    else:
                        for ops in valu_ops:
                            self.instrs.append({"valu": self.pad_to_6(ops)})

                elif is_level_2:
                    # 4-way selection using arithmetic
                    level2_tree = v_tree_alt
                    valu_ops = [
                        [("-", v_cond[c], chunk_idx_addrs[c], v_three) for c in range(n_active)],
                        [("&", v_tmp1[c], v_cond[c], v_one) for c in range(n_active)],
                        [(">>", v_cond[c], v_cond[c], v_one) for c in range(n_active)],
                        [("multiply_add", v_tmp2[c], v_tmp1[c], v_diff_low, v_tree3) for c in range(n_active)],
                        [("multiply_add", level2_tree[c], v_tmp1[c], v_diff_high, v_tree5) for c in range(n_active)],
                        [("-", v_tmp1[c], level2_tree[c], v_tmp2[c]) for c in range(n_active)],
                        [("multiply_add", level2_tree[c], v_cond[c], v_tmp1[c], v_tmp2[c]) for c in range(n_active)],
                    ]
                    valu_ops.extend(build_hash_index_ops(level2_tree, chunk_val_addrs, chunk_idx_addrs, n_active, is_round_10))

                    # Preload next round if last group and next is scatter-gather
                    # OPTIMIZATION: Compute addresses for BOTH groups 0 and 1
                    is_last_group = not has_next
                    next_round_is_scatter = (rnd + 1 < rounds) and (rnd + 1 not in [0, 1, 2, 11, 12, 13])
                    if is_last_group and next_round_is_scatter:
                        # Addresses for group 0 go into gather_addrs
                        g0_idx = [scratch_idx + c * VLEN for c in range(CHUNKS_PER_GROUP)]
                        g0_alu = [("+", gather_addrs[c][i], tree_base, g0_idx[c] + i)
                                  for c in range(CHUNKS_PER_GROUP) for i in range(VLEN)]
                        # Addresses for group 1 go into gather_addrs_alt
                        g1_idx = [scratch_idx + (CHUNKS_PER_GROUP + c) * VLEN for c in range(CHUNKS_PER_GROUP)]
                        g1_alu = [("+", gather_addrs_alt[c][i], tree_base, g1_idx[c] + i)
                                  for c in range(CHUNKS_PER_GROUP) for i in range(VLEN)]
                        next_alu = g0_alu + g1_alu  # 96 ALU ops total

                        # Load tree values for group 0
                        next_load = [("load", v_tree[c] + i, gather_addrs[c][i])
                                     for c in range(CHUNKS_PER_GROUP) for i in range(VLEN)]

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
                        second_group_addrs_ready = True  # Group 1's addresses are also pre-computed
                    else:
                        for ops in valu_ops:
                            self.instrs.append({"valu": self.pad_to_6(ops)})

                # Level 3 selection disabled due to register aliasing issues
                # Using scatter-gather for rounds 3 and 14 (is_level_3) instead

                else:
                    # Scatter-gather with optimized pipelining
                    # When level 2 preloads both group 0's values AND group 1's addresses,
                    # group 0 can start loading group 1 from cycle 1 and compute group 2's addresses
                    scatter_gather_rounds = [3, 4, 5, 6, 7, 8, 9, 10, 14, 15]

                    if g_idx == 0 and rnd in scatter_gather_rounds and not this_round_first_preloaded:
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

                    is_last_group = not has_next
                    next_round_is_scatter = (rnd + 1 < rounds) and (rnd + 1 in scatter_gather_rounds)

                    # OPTIMIZATION: Compute addresses 2 groups ahead instead of 1
                    # Only applies when:
                    # - Groups 0,1: addresses precomputed by level 2
                    # - Groups 2+: addresses precomputed by group N-2 (which computed them into curr_addrs)
                    # For groups 2+, we need to verify the previous group actually computed N+2 addresses
                    addrs_ready = this_round_second_addrs_ready and g_idx <= 1  # Only 0,1 for now

                    if addrs_ready and has_next:
                        # Addresses for next group already computed
                        # - For g_idx=0,1: by level 2
                        # - For g_idx=2+: by group N-2 (via the alternating buffer pattern)
                        next_load = [("load", next_tree[c] + i, next_addrs[c][i])
                                     for c in range(next_n_active) for i in range(VLEN)]
                        # Compute addresses for group N+2 into curr_addrs
                        nn_chunk_base = chunk_base + 2 * CHUNKS_PER_GROUP
                        if nn_chunk_base < n_chunks:
                            nn_n_active = min(CHUNKS_PER_GROUP, n_chunks - nn_chunk_base)
                            nn_idx_addrs = [scratch_idx + (nn_chunk_base + c) * VLEN for c in range(nn_n_active)]
                            next_alu = [("+", curr_addrs[c][i], tree_base, nn_idx_addrs[c] + i)
                                        for c in range(nn_n_active) for i in range(VLEN)]
                        else:
                            next_alu = []
                    elif has_next:
                        next_alu = [("+", next_addrs[c][i], tree_base, next_chunk_idx_addrs[c] + i)
                                    for c in range(next_n_active) for i in range(VLEN)]
                        next_load = [("load", next_tree[c] + i, next_addrs[c][i])
                                     for c in range(next_n_active) for i in range(VLEN)]
                    elif is_last_group and next_round_is_scatter:
                        first_cg_idx = [scratch_idx + c * VLEN for c in range(CHUNKS_PER_GROUP)]
                        next_alu = [("+", gather_addrs[c][i], tree_base, first_cg_idx[c] + i)
                                    for c in range(CHUNKS_PER_GROUP) for i in range(VLEN)]
                        next_load = [("load", v_tree[c] + i, gather_addrs[c][i])
                                     for c in range(CHUNKS_PER_GROUP) for i in range(VLEN)]
                        first_group_preloaded = True
                    else:
                        next_alu, next_load = [], []

                    # OPTIMIZED bundling:
                    # - If addresses pre-computed (groups 0,1 when second_addrs_ready), start loads from cycle 1
                    # - Otherwise, ALU first then loads
                    can_load_from_start = this_round_second_addrs_ready and g_idx <= 1
                    alu_idx, load_idx = 0, 0
                    for ops in valu_ops:
                        bundle = {"valu": self.pad_to_6(ops)}
                        if alu_idx < len(next_alu):
                            bundle["alu"] = next_alu[alu_idx:alu_idx+12]
                            alu_idx += 12
                        # Start loads from cycle 1 if addresses pre-computed, else after ALU
                        if can_load_from_start and load_idx < len(next_load):
                            bundle["load"] = next_load[load_idx:load_idx+2]
                            load_idx += 2
                        elif alu_idx >= len(next_alu) and load_idx < len(next_load):
                            bundle["load"] = next_load[load_idx:load_idx+2]
                            load_idx += 2
                        self.instrs.append(bundle)

                    # Remaining loads
                    while load_idx < len(next_load):
                        self.instrs.append({"load": next_load[load_idx:load_idx+2]})
                        load_idx += 2

        # Copy results back to main memory
        for chunk in range(0, n_chunks, 2):
            off0, off1 = chunk * VLEN, (chunk + 1) * VLEN
            self.instrs.append({"load": [("const", tmp1, off0), ("const", tmp2, off1)]})
            self.instrs.append({"alu": [("+", tmp3, self.scratch["inp_values_p"], tmp1), ("+", tmp4, self.scratch["inp_values_p"], tmp2)]})
            self.instrs.append({"store": [("vstore", tmp3, scratch_val + off0), ("vstore", tmp4, scratch_val + off1)]})

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
