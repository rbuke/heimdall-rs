use hashbrown::HashMap;

use alloy::primitives::U256;
use heimdall_common::constants::{MEMORY_REGEX, STORAGE_REGEX};
use tracing::trace;

use crate::core::{
    opcodes::{JUMPDEST, PUSH0, PUSH1, PUSH32},
    stack::{Stack, StackFrame},
};

use super::jump_frame::JumpFrame;

/// For every byte offset of the bytecode, whether it holds a JUMPDEST opcode (push data that
/// happens to be 0x5b is not one).
pub(super) fn jumpdest_table(bytecode: &[u8]) -> Vec<bool> {
    let mut table = vec![false; bytecode.len()];
    let mut i = 0;
    while i < bytecode.len() {
        let opcode = bytecode[i];
        if opcode == JUMPDEST {
            table[i] = true;
        }
        i += 1;
        if (PUSH1..=PUSH32).contains(&opcode) {
            i += (opcode - PUSH1 + 1) as usize;
        }
    }
    table
}

/// The call context of a stack: the return addresses on it, bottom to top. heimdall folds every
/// constant into a PUSH, so a return address is a constant frame whose value is a JUMPDEST
/// offset. Two stacks with different return addresses are in different internal-function calls.
pub(super) fn call_context(stack: &Stack, jumpdests: &[bool]) -> Vec<U256> {
    stack
        .stack
        .iter()
        .rev()
        .filter(|frame| (PUSH0..=PUSH32).contains(&frame.operation.opcode))
        .map(|frame| frame.value)
        .filter(|value| value.lt(&U256::from(jumpdests.len())) && jumpdests[value.to::<usize>()])
        .collect()
}

/// Given two stacks A and B, return A - B, i.e. the items in A that are not in B.
/// This operation takes order into account, so if A = [1, 2, 3] and B = [1, 3, 2], then A - B =
/// [2]. This is referred to as the "stack diff"
pub(super) fn stack_diff(a: &Stack, b: &Stack) -> Vec<StackFrame> {
    let mut diff = Vec::new();

    for (i, frame) in a.stack.iter().enumerate() {
        if b.stack.len() <= i || frame != &b.stack[i] {
            diff.push(frame.clone());
        }
    }

    diff
}

/// Like [`stack_diff`], but keeps only the slots of A whose expression grew out of the slot at the
/// same position in B (B's expression is a strict sub-expression of A's, e.g. `i` -> `i + 1`), plus
/// slots B does not have. A slot that was merely replaced by an unrelated value is left out.
pub(super) fn grown_stack_diff(a: &Stack, b: &Stack) -> Vec<StackFrame> {
    let mut diff = Vec::new();

    for (i, frame) in a.stack.iter().enumerate() {
        if b.stack.len() <= i {
            diff.push(frame.clone());
            continue;
        }
        if frame == &b.stack[i] {
            continue;
        }
        let new = frame.operation.solidify();
        let old = b.stack[i].operation.solidify();
        if new.len() > old.len() && new.contains(&old) {
            diff.push(frame.clone());
        }
    }

    diff
}

/// Check if the given stack contains too many items to feasibly
/// reach the bottom of the stack without being a loop.
pub(super) fn stack_contains_too_many_items(stack: &Stack) -> bool {
    if stack.size() > 320 {
        // 320 is an arbitrary number, i picked it randomly :D
        trace!("jump matches loop-detection heuristic: 'stack_contains_too_many_items'",);
        return true;
    }

    false
}

/// Check if the current jump frame has a stack depth less than the max stack depth of all previous
/// matching jumps. If yes, the stack is not growing and we likely have a loop.
///
/// Only previous jumps made in the same call context (the same return addresses on the stack,
/// see [`call_context`]) count. A shared helper such as solc's memory allocator is first reached
/// deep inside one caller and later from a shallower call site; the later call is a new call,
/// not a loop, and pruning it dropped everything behind it.
pub(super) fn jump_stack_depth_less_than_max_stack_depth(
    current_jump_frame: &JumpFrame,
    current_context: &[U256],
    handled_jumps: &HashMap<JumpFrame, Vec<Stack>>,
    jumpdests: &[bool],
) -> bool {
    // (1) get all keys that match current_jump_frame.pc and current_jump_frame.jumpdest and were
    // seen in the current call context
    let matching_keys = handled_jumps
        .iter()
        .filter(|(key, stacks)| {
            key.pc == current_jump_frame.pc &&
                key.jumpdest == current_jump_frame.jumpdest &&
                stacks.iter().any(|stack| call_context(stack, jumpdests) == current_context)
        })
        .map(|(key, _)| key)
        .collect::<Vec<&JumpFrame>>();

    // (a) get the max stack_depth of all matching keys
    let max_stack_depth = matching_keys.iter().map(|key| key.stack_depth).max().unwrap_or(0);

    // (b) if the current stack depth is less than the max stack depth, we don't need to
    // continue.
    if current_jump_frame.stack_depth < max_stack_depth {
        trace!(
            "jump matches loop-detection heuristic: 'jump_stack_depth_less_than_max_stack_depth'"
        );
        trace!("jump terminated.");
        return true;
    }

    false
}

/// Check if the given stack contains too many of the same item.
/// If the stack contains more than 16 of the same item (with the same sources), it is considered a
/// loop.
pub(super) fn stack_contains_too_many_of_the_same_item(stack: &Stack) -> bool {
    if stack.size() <= 16 {
        return false;
    }

    // solidify each frame once: doing it per pair made this quadratic in the stack size, and the
    // check runs at every JUMPI
    let mut counts: HashMap<String, usize> = HashMap::new();
    for frame in &stack.stack {
        *counts.entry(frame.operation.solidify()).or_default() += 1;
    }
    if counts.values().any(|count| *count >= 16) {
        trace!("jump matches loop-detection heuristic: 'stack_contains_too_many_of_the_same_item'",);
        return true;
    }

    false
}

/// Check if the stack contains any item with a source operation depth > 16. If so, it is considered
/// a loop. This check originates from the `stack too deep` error in Solidity due to the `DUP16` and
/// `SWAP16` operation limitations.
pub(super) fn stack_item_source_depth_too_deep(stack: &Stack) -> bool {
    if stack.stack.iter().any(|frame| frame.operation.depth() > 16) {
        trace!("jump matches loop-detection heuristic: 'stack_item_source_depth_too_deep'");
        return true;
    }

    false
}

/// Compare the stack diff to the given jump condition and determine if the jump condition appears
/// to be the condition of a loop.
pub(super) fn jump_condition_appears_recursive(
    stack_diff: &[StackFrame],
    jump_condition: &str,
) -> bool {
    // check if the jump condition appears in the stack diff more than once, this is likely a loop
    if stack_diff
        .iter()
        .map(|frame| frame.operation.solidify())
        .any(|solidified| jump_condition.contains(&solidified))
    {
        trace!("jump matches loop-detection heuristic: 'jump_condition_appears_recursive'");
        return true;
    }

    false
}

/// Check if the jump condition contains a memory access that is modified within the stack diff.
pub(super) fn jump_condition_contains_mutated_memory_access(
    stack_diff: &[StackFrame],
    jump_condition: &str,
) -> bool {
    let mut memory_accesses = MEMORY_REGEX.find_iter(jump_condition);
    if stack_diff.iter().any(|frame| {
        memory_accesses.any(|_match| {
            if _match.is_err() {
                return false;
            }

            let memory_access = match _match {
                Ok(access) => access,
                Err(_) => return false,
            };

            let slice = &jump_condition[memory_access.start()..memory_access.end()];
            frame.operation.solidify().contains(slice)
        })
    }) {
        trace!("jump matches loop-detection heuristic: 'jump_condition_contains_mutated_memory_access'");
        return true;
    }

    false
}

/// Check if the jump condition contains a storage access that is modified within the stack diff.
pub(super) fn jump_condition_contains_mutated_storage_access(
    stack_diff: &[StackFrame],
    jump_condition: &str,
) -> bool {
    let mut storage_accesses = STORAGE_REGEX.find_iter(jump_condition);
    if stack_diff.iter().any(|frame| {
        storage_accesses.any(|_match| {
            if _match.is_err() {
                return false;
            }
            let storage_access = match _match {
                Ok(access) => access,
                Err(_) => return false,
            };
            let slice = &jump_condition[storage_access.start()..storage_access.end()];
            frame.operation.solidify().contains(slice)
        })
    }) {
        trace!("jump matches loop-detection heuristic: 'jump_condition_contains_mutated_storage_access'");
        return true;
    }

    false
}

/// check if all stack diffs for all historical stacks are exactly length 1, and the same
pub(super) fn historical_diffs_approximately_equal(
    stack: &Stack,
    historical_stacks: &[Stack],
) -> bool {
    // break if historical_stacks.len() < 4
    // this is an arbitrary number, i picked it randomly :D
    if historical_stacks.len() < 4 {
        return false;
    }

    // get the stack diffs for all historical stacks
    let mut stack_diffs = Vec::new();
    for historical_stack in historical_stacks {
        stack_diffs.push(
            stack_diff(stack, historical_stack)
                .iter()
                .map(|frame| frame.value)
                .collect::<Vec<U256>>(),
        );
    }

    // get stack length / 10, rounded up as threshold
    let threshold = (stack.size() as f64 / 10f64).ceil() as usize;

    // check if all stack diffs are similar
    if !stack_diffs.iter().all(|diff| diff.len() <= threshold) {
        return false;
    }

    // check if all stack diffs are the same
    if !stack_diffs
        .iter()
        .all(|diff| diff.first() == stack_diffs.first().unwrap_or(&vec![]).first())
    {
        return false;
    }

    trace!("jump matches loop-detection heuristic: 'jump_condition_historical_diffs_approximately_equal'");

    true
}

/// Check if any stack position shows a consistent pattern (increasing, decreasing, or alternating)
/// across more than 32 iterations. This indicates a loop counter or iterator.
pub(super) fn stack_position_shows_pattern(stack: &Stack, historical_stacks: &[Stack]) -> bool {
    // Start checking after just 10 iterations to catch loops earlier
    if historical_stacks.len() < 10 {
        return false;
    }

    trace!("checking stack pattern with {} historical stacks", historical_stacks.len());

    // Determine the maximum stack size to check all positions
    let max_size = stack.size().max(historical_stacks.iter().map(|s| s.size()).max().unwrap_or(0));

    // For each stack position, collect values across all historical stacks
    for position in 0..max_size {
        let mut values: Vec<U256> = Vec::new();

        // Collect values at this position from all historical stacks
        for hist_stack in historical_stacks {
            if let Some(frame) = hist_stack.stack.get(position) {
                values.push(frame.value);
            }
        }

        // Also include current stack value
        if let Some(frame) = stack.stack.get(position) {
            values.push(frame.value);
        }

        // Need at least 10 values to detect a meaningful pattern
        if values.len() < 10 {
            continue;
        }

        // Check for patterns
        if is_consistently_increasing(&values) {
            trace!(
                "jump matches loop-detection heuristic: 'stack_position_shows_pattern' \
                 (increasing at position {})",
                position
            );
            return true;
        }

        if is_consistently_decreasing(&values) {
            trace!(
                "jump matches loop-detection heuristic: 'stack_position_shows_pattern' \
                 (decreasing at position {})",
                position
            );
            return true;
        }

        if is_consistently_alternating(&values) {
            trace!(
                "jump matches loop-detection heuristic: 'stack_position_shows_pattern' \
                 (alternating at position {})",
                position
            );
            return true;
        }
    }

    false
}

/// Check if values show a strong increasing trend (>= 60% of pairs increase)
fn is_consistently_increasing(values: &[U256]) -> bool {
    if values.len() < 2 {
        return false;
    }

    let total_pairs = values.len() - 1;
    let increasing_pairs = values.windows(2).filter(|pair| pair[1] > pair[0]).count();

    // Require at least 60% of pairs to be increasing
    increasing_pairs as f64 / total_pairs as f64 >= 0.6
}

/// Check if values show a strong decreasing trend (>= 60% of pairs decrease)
fn is_consistently_decreasing(values: &[U256]) -> bool {
    if values.len() < 2 {
        return false;
    }

    let total_pairs = values.len() - 1;
    let decreasing_pairs = values.windows(2).filter(|pair| pair[1] < pair[0]).count();

    // Require at least 60% of pairs to be decreasing
    decreasing_pairs as f64 / total_pairs as f64 >= 0.6
}

/// Check if values show a strong alternating pattern (>= 60% of triples alternate)
fn is_consistently_alternating(values: &[U256]) -> bool {
    if values.len() < 3 {
        return false;
    }

    let total_triples = values.len() - 2;
    let alternating_triples = values
        .windows(3)
        .filter(|triple| {
            (triple[1] > triple[0] && triple[2] < triple[1]) ||
                (triple[1] < triple[0] && triple[2] > triple[1])
        })
        .count();

    // Require at least 60% of triples to alternate
    alternating_triples as f64 / total_triples as f64 >= 0.6
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use alloy::primitives::U256;

    use hashbrown::HashMap;

    use super::{
        call_context, grown_stack_diff, jump_condition_appears_recursive,
        jump_stack_depth_less_than_max_stack_depth, jumpdest_table,
    };
    use crate::{
        core::{
            opcodes::{WrappedInput, WrappedOpcode, ADD, CALLDATALOAD, PUSH2},
            stack::Stack,
        },
        ext::exec::jump_frame::JumpFrame,
    };

    fn arg(offset: u64) -> WrappedOpcode {
        WrappedOpcode::new(CALLDATALOAD, vec![WrappedInput::Raw(U256::from(offset))])
    }

    fn constant(value: u64) -> WrappedOpcode {
        WrappedOpcode::new(PUSH2, vec![WrappedInput::Raw(U256::from(value))])
    }

    /// `PUSH2 0x005b JUMP JUMPDEST STOP JUMPDEST STOP`: JUMPDESTs at 4 and 6, and the 0x5b byte
    /// inside the push data at offset 2 is not one
    fn bytecode() -> Vec<u8> {
        vec![0x61, 0x00, 0x5b, 0x56, 0x5b, 0x00, 0x5b, 0x00]
    }

    #[test]
    fn test_jumpdest_table_skips_push_data() {
        assert_eq!(
            jumpdest_table(&bytecode()),
            vec![false, false, false, false, true, false, true, false]
        );
    }

    #[test]
    fn test_call_context_is_the_return_addresses() {
        let jumpdests = jumpdest_table(&bytecode());
        let mut stack = Stack::new();
        stack.push(U256::from(4), constant(4)); // return address
        stack.push(U256::from(1), arg(4)); // an argument
        stack.push(U256::from(2), constant(2)); // a constant that is not a JUMPDEST offset
        stack.push(U256::from(6), arg(36)); // an argument that happens to equal a JUMPDEST offset
        stack.push(U256::from(6), constant(6)); // a nested return address
        assert_eq!(call_context(&stack, &jumpdests), vec![U256::from(4), U256::from(6)]);
    }

    #[test]
    fn test_depth_check_only_compares_the_same_call_context() {
        let jumpdests = jumpdest_table(&bytecode());
        let mut handled_jumps = HashMap::new();

        let frame = |stack: &Stack| JumpFrame::new(10, U256::from(20), stack.size(), true);
        let check = |stack: &Stack, handled_jumps: &HashMap<JumpFrame, Vec<Stack>>| {
            jump_stack_depth_less_than_max_stack_depth(
                &frame(stack),
                &call_context(stack, &jumpdests),
                handled_jumps,
                &jumpdests,
            )
        };

        // a shared helper's JUMPI first reached 3 deep from the call site returning to 4
        let mut deep = Stack::new();
        deep.push(U256::from(4), constant(4));
        deep.push(U256::from(1), arg(4));
        deep.push(U256::from(2), arg(36));
        handled_jumps.insert(frame(&deep), vec![deep]);

        // the same helper reached 1 deep from the call site returning to 6: a new call
        let mut shallow = Stack::new();
        shallow.push(U256::from(6), constant(6));
        assert!(!check(&shallow, &handled_jumps));

        // reached 1 deep again from the first call site: the stack shrank within one call
        let mut shrunk = Stack::new();
        shrunk.push(U256::from(4), constant(4));
        assert!(check(&shrunk, &handled_jumps));
    }

    #[test]
    fn test_grown_stack_diff_ignores_replaced_slot() {
        // one shared validator called for arg0, then for arg1: the slot is replaced, not grown
        let (mut first, mut second) = (Stack::new(), Stack::new());
        first.push(U256::from(1), arg(4));
        second.push(U256::from(2), arg(36));
        let diff = grown_stack_diff(&second, &first);
        assert!(diff.is_empty());
        assert!(!jump_condition_appears_recursive(&diff, "arg1 == (address(arg1))"));
    }

    #[test]
    fn test_grown_stack_diff_keeps_grown_slot() {
        // a loop counter: `arg0` -> `arg0 + 0x01`
        let (mut first, mut second) = (Stack::new(), Stack::new());
        first.push(U256::from(1), arg(4));
        second.push(
            U256::from(2),
            WrappedOpcode::new(
                ADD,
                vec![WrappedInput::Opcode(Arc::new(arg(4))), WrappedInput::Raw(U256::from(1))],
            ),
        );
        let diff = grown_stack_diff(&second, &first);
        assert_eq!(diff.len(), 1);
    }
}
