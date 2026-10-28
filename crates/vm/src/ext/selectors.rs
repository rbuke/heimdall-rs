use hashbrown::{HashMap, HashSet};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use alloy::primitives::U256;
use eyre::Result;
use heimdall_common::{
    ether::signatures::{ResolveSelector, ResolvedFunction},
    utils::strings::decode_hex,
};
use tokio::task;
use tracing::{debug, error, info, trace, warn};

use crate::core::{
    opcodes::{
        WrappedInput, WrappedOpcode, CALLDATALOAD, EQ, ISZERO, JUMPI, MLOAD, PUSH0, PUSH32, SUB,
        XOR,
    },
    vm::VM,
};

/// Finds and resolves function selectors from disassembled bytecode
///
/// This function analyzes disassembled EVM bytecode to extract function selectors
/// and optionally resolves them to human-readable function signatures.
///
/// # Arguments
/// * `disassembled_bytecode` - The disassembled EVM bytecode to analyze
/// * `skip_resolving` - If true, skip the process of resolving selectors to function signatures
/// * `evm` - The VM instance to use for analysis
///
/// # Returns
/// * A Result containing a tuple with:
///   - A HashMap mapping selector strings to their instruction offsets
///   - A HashMap mapping selector strings to their resolved function information
pub async fn get_resolved_selectors(
    disassembled_bytecode: &str,
    skip_resolving: &bool,
    evm: &VM,
) -> Result<(HashMap<String, u128>, HashMap<String, Vec<ResolvedFunction>>)> {
    let selectors = find_function_selectors(evm, disassembled_bytecode);

    let mut resolved_selectors = HashMap::new();
    if !skip_resolving {
        resolved_selectors =
            resolve_selectors::<ResolvedFunction>(selectors.keys().cloned().collect()).await;

        trace!(
            "resolved {} possible functions from {} detected selectors.",
            resolved_selectors.len(),
            selectors.len()
        );
    } else {
        trace!("found {} possible function selectors.", selectors.len());
    }

    Ok((selectors, resolved_selectors))
}

/// find all function selectors in the given EVM assembly.
///
/// Every PUSH4 is a candidate, and so is a PUSH1 to PUSH3 the dispatcher compares (a selector with
/// leading zero bytes compiles to the shorter push, `0x00fdd58e` is `PUSH3 fdd58e`). Each
/// candidate is then resolved from the control flow by [`resolve_entry_point`].
pub fn find_function_selectors(evm: &VM, assembly: &str) -> HashMap<String, u128> {
    let mut function_selectors = HashMap::new();
    let mut handled_selectors = HashSet::new();

    // search through assembly for PUSHN (where N <= 4) instructions, optimistically assuming that
    // they are function selectors
    let assembly: Vec<Vec<String>> = assembly
        .split('\n')
        .map(|line| line.trim().split(' ').map(|arg| arg.to_string()).collect())
        .collect();
    for (index, instruction_args) in assembly.iter().enumerate() {
        if instruction_args.len() < 3 {
            continue;
        }

        let width = match instruction_args[1].as_str() {
            "PUSH4" => 4,
            "PUSH3" => 3,
            "PUSH2" => 2,
            "PUSH1" => 1,
            _ => continue,
        };
        if width < 4 && !push_is_compared(&assembly, index) {
            continue;
        }

        // zero-pad a short push to the 4-byte selector it stands for
        let function_selector = format!("{:0>8}", instruction_args[2]);

        // check if this function selector has already been handled
        if handled_selectors.contains(&function_selector) {
            continue;
        }

        trace!(
            "optimistically assuming instruction {} {} {} is a function selector",
            instruction_args[0],
            instruction_args[1],
            instruction_args[2]
        );

        // add the function selector to the handled selectors
        handled_selectors.insert(function_selector.clone());

        // get the function's entry point
        let function_entry_point = match resolve_entry_point(&mut evm.clone(), &function_selector) {
            0 => continue,
            x => x,
        };

        trace!(
            "found function selector {} at entry point {}",
            function_selector,
            function_entry_point
        );

        function_selectors.insert(function_selector, function_entry_point);
    }

    info!("discovered {} function selectors in assembly", function_selectors.len());
    function_selectors
}

/// Whether the push at `index` of the assembly feeds a compare: it is followed by an optional
/// DUP, an optional `PUSH1 00 MLOAD` (vyper 0.2 reloads the selector from memory) and then EQ,
/// SUB, XOR, LT or GT. This only screens short pushes as candidates; the entry point itself is
/// resolved by executing the dispatcher.
fn push_is_compared(assembly: &[Vec<String>], index: usize) -> bool {
    let op = |i: usize| assembly.get(i).and_then(|args| args.get(1)).map_or("", String::as_str);
    let arg = |i: usize| assembly.get(i).and_then(|args| args.get(2)).map_or("", String::as_str);

    let mut i = index + 1;
    if op(i).starts_with("DUP") {
        i += 1;
    }
    if op(i) == "PUSH1" && arg(i) == "00" && op(i + 1) == "MLOAD" {
        i += 2;
    }
    matches!(op(i), "EQ" | "SUB" | "XOR" | "LT" | "GT")
}

/// resolve a selector's function entry point from the EVM bytecode
///
/// Runs the dispatcher with the selector as calldata. The JUMPI whose condition compares the
/// call's selector against this one decides the entry point: the body starts where the matching
/// branch goes. That is the jump target for a solc `EQ` leaf, and the instruction after the JUMPI
/// for the forms that jump away on a mismatch and fall through into the body: the `SUB` last leaf
/// of a solc table, vyper 0.3's `XOR` and vyper 0.2's `EQ ISZERO`. Returns 0 when no compare
/// matches, i.e. the dispatcher does not answer this selector.
pub fn resolve_entry_point(vm: &mut VM, selector: &str) -> u128 {
    let mut handled_jumps = HashSet::new();
    let selector_value = match U256::from_str_radix(selector, 16) {
        Ok(value) => value,
        Err(_) => return 0,
    };

    // execute the EVM call to find the entry point for the given selector
    vm.calldata = decode_hex(selector).expect("Failed to decode selector.");
    while vm.bytecode.len() >= vm.instruction as usize {
        let call = match vm.step() {
            Ok(call) => call,
            Err(_) => break, // the call failed, so we can't resolve the selector
        };

        if call.last_instruction.opcode == JUMPI {
            let jump_taken = !call.last_instruction.inputs[1].is_zero();
            let jumpdest: u128 = call.last_instruction.inputs[0].try_into().unwrap_or(0);

            // the branch the dispatcher took is the matching one exactly when the compare's
            // other side is the call's selector, so a compare taken the other way is not this
            // selector's leaf
            if let Some(taken_on_match) =
                selector_compare(&call.last_instruction.input_operations[1], selector_value)
            {
                if jump_taken == taken_on_match {
                    return if jump_taken {
                        jumpdest
                    } else {
                        // `instruction` is the 1-based offset of the JUMPI, i.e. the 0-based
                        // offset of the instruction after it
                        call.last_instruction.instruction
                    };
                }
            }

            // loops aren't supported in the dispatcher, so a jump target seen twice means the
            // walk left the dispatcher without a match
            if jump_taken && !handled_jumps.insert(jumpdest) {
                return 0;
            }
        }

        if vm.exitcode != 255 || !vm.returndata.is_empty() {
            break;
        }
    }

    0
}

/// Whether `condition`, a JUMPI condition, is an exact compare of the call's selector against
/// `selector`, and if so whether the jump is taken on a match: `Some(true)` for `EQ`, `Some(false)`
/// for `SUB`, `XOR` and `EQ ISZERO`, each with any further `ISZERO` wrapping folded in. `None`
/// for anything else, including the `LT`/`GT` pivots of a binary-search table.
///
/// The side that is not the constant has to read the call's input: the calldata, or the memory
/// vyper 0.2 copies the selector to.
fn selector_compare(condition: &WrappedOpcode, selector: U256) -> Option<bool> {
    let mut negated = false;
    let mut op = condition;
    while op.opcode == ISZERO {
        negated = !negated;
        op = match op.inputs.first() {
            Some(WrappedInput::Opcode(inner)) => inner,
            _ => return None,
        };
    }

    let taken_on_match = match op.opcode {
        EQ => true,
        SUB | XOR => false,
        _ => return None,
    };

    let [a, b] = op.inputs.as_slice() else { return None };
    let other = if push_value(a) == Some(selector) {
        b
    } else if push_value(b) == Some(selector) {
        a
    } else {
        return None;
    };

    match other {
        WrappedInput::Opcode(inner) if reads_input(inner) => Some(taken_on_match != negated),
        _ => None,
    }
}

/// The constant a push (or a constant heimdall folded into a PUSH32) holds.
fn push_value(input: &WrappedInput) -> Option<U256> {
    match input {
        WrappedInput::Raw(value) => Some(*value),
        WrappedInput::Opcode(op) if (PUSH0..=PUSH32).contains(&op.opcode) => {
            match op.inputs.first() {
                Some(WrappedInput::Raw(value)) => Some(*value),
                None => Some(U256::ZERO),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Whether the operation reads the calldata or memory anywhere in its expression tree.
fn reads_input(op: &WrappedOpcode) -> bool {
    op.opcode == CALLDATALOAD ||
        op.opcode == MLOAD ||
        op.inputs
            .iter()
            .any(|input| matches!(input, WrappedInput::Opcode(inner) if reads_input(inner)))
}

/// Resolve a list of selectors to their function signatures.
pub async fn resolve_selectors<T>(selectors: Vec<String>) -> HashMap<String, Vec<T>>
where
    T: ResolveSelector + Send + Clone + 'static, {
    // short-circuit if there are no selectors
    if selectors.is_empty() {
        return HashMap::new();
    }

    let resolved_functions: Arc<Mutex<HashMap<String, Vec<T>>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let mut threads = Vec::new();
    let start_time = Instant::now();
    let selector_count = selectors.len();

    for selector in selectors {
        let function_clone = resolved_functions.clone();

        // create a new thread for each selector
        threads.push(task::spawn(async move {
            if let Ok(Some(function)) = T::resolve(&selector).await {
                let mut _resolved_functions =
                    function_clone.lock().expect("Could not obtain lock on function_clone.");
                _resolved_functions.insert(selector, function);
            }
        }));
    }

    // wait for all threads to finish
    for thread in threads {
        if let Err(e) = thread.await {
            // Handle error
            error!("failed to resolve selector: {:?}", e);
        }
    }

    let signatures =
        resolved_functions.lock().expect("failed to obtain lock on resolved_functions.").clone();
    if signatures.is_empty() {
        warn!("failed to resolve any signatures from {} selectors", selector_count);
    }
    info!("resolved {} signatures from {} selectors", signatures.len(), selector_count);
    debug!("signature resolution took {:?}", start_time.elapsed());
    signatures
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use alloy::primitives::U256;

    use super::{push_is_compared, selector_compare};
    use crate::core::opcodes::{
        WrappedInput, WrappedOpcode, CALLDATALOAD, EQ, GT, ISZERO, MLOAD, PUSH0, PUSH1, PUSH4, SHR,
        SLOAD, SUB, XOR,
    };

    const SELECTOR: u64 = 0xf04f2707;

    fn op(opcode: u8, inputs: Vec<WrappedOpcode>) -> WrappedOpcode {
        WrappedOpcode::new(
            opcode,
            inputs.into_iter().map(|input| WrappedInput::Opcode(Arc::new(input))).collect(),
        )
    }

    fn push(opcode: u8, value: u64) -> WrappedOpcode {
        WrappedOpcode::new(opcode, vec![WrappedInput::Raw(U256::from(value))])
    }

    /// `msg.data[0] >> 0xe0`, the selector as solc and vyper 0.3 compute it
    fn msg_sig() -> WrappedOpcode {
        op(SHR, vec![push(PUSH1, 0xe0), op(CALLDATALOAD, vec![push(PUSH0, 0)])])
    }

    fn assembly(lines: &[&str]) -> Vec<Vec<String>> {
        lines.iter().map(|line| line.split(' ').map(str::to_string).collect()).collect()
    }

    #[test]
    fn test_selector_compare_eq_leaf_jumps_on_match() {
        let condition = op(EQ, vec![msg_sig(), push(PUSH4, SELECTOR)]);
        assert_eq!(selector_compare(&condition, U256::from(SELECTOR)), Some(true));
    }

    #[test]
    fn test_selector_compare_fallthrough_forms_jump_on_mismatch() {
        // the last leaf of a solc table, vyper 0.3, vyper 0.2 (selector reloaded from memory)
        let sub = op(SUB, vec![push(PUSH4, SELECTOR), msg_sig()]);
        let xor = op(XOR, vec![push(PUSH4, SELECTOR), msg_sig()]);
        let eq_iszero =
            op(ISZERO, vec![op(EQ, vec![push(PUSH4, SELECTOR), op(MLOAD, vec![push(PUSH1, 0)])])]);
        for condition in [sub, xor, eq_iszero] {
            assert_eq!(selector_compare(&condition, U256::from(SELECTOR)), Some(false));
        }

        // a negated mismatch test jumps on a match again
        let iszero_sub = op(ISZERO, vec![op(SUB, vec![push(PUSH4, SELECTOR), msg_sig()])]);
        assert_eq!(selector_compare(&iszero_sub, U256::from(SELECTOR)), Some(true));
    }

    #[test]
    fn test_selector_compare_ignores_pivots_other_selectors_and_non_input_reads() {
        let pivot = op(GT, vec![push(PUSH4, SELECTOR), msg_sig()]);
        let other_selector = op(EQ, vec![msg_sig(), push(PUSH4, 0x610c95d0)]);
        let storage = op(EQ, vec![op(SLOAD, vec![push(PUSH1, 0)]), push(PUSH4, SELECTOR)]);
        for condition in [pivot, other_selector, storage] {
            assert_eq!(selector_compare(&condition, U256::from(SELECTOR)), None);
        }
    }

    #[test]
    fn test_push_is_compared() {
        // vyper 0.3: PUSH3 007c98ab DUP2 XOR
        let vyper3 = assembly(&["000000 PUSH3 7c98ab", "000004 DUP2 ", "000005 XOR "]);
        assert!(push_is_compared(&vyper3, 0));

        // vyper 0.2: PUSH3 00fdd58e PUSH1 00 MLOAD EQ ISZERO
        let vyper2 = assembly(&[
            "000000 PUSH3 fdd58e",
            "000004 PUSH1 00",
            "000006 MLOAD ",
            "000007 EQ ",
            "000008 ISZERO ",
        ]);
        assert!(push_is_compared(&vyper2, 0));

        // solc: PUSH3 fdd58e EQ
        let solc = assembly(&["000000 PUSH3 fdd58e", "000004 EQ "]);
        assert!(push_is_compared(&solc, 0));

        // arithmetic and jump targets are not selectors
        let add = assembly(&["000000 PUSH1 20", "000002 ADD "]);
        let jump = assembly(&["000000 PUSH2 0243", "000003 JUMP "]);
        let end = assembly(&["000000 PUSH1 20"]);
        for (asm, index) in [(add, 0), (jump, 0), (end, 0)] {
            assert!(!push_is_compared(&asm, index));
        }
    }
}
