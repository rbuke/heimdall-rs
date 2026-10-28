//! Tells a public getter's return apart from a computed value.

use fancy_regex::Regex;
use heimdall_common::utils::strings::find_balanced_encapsulator;
use lazy_static::lazy_static;

use crate::utils::constants::{SCRATCH_ASSIGNMENT_REGEX, STORAGE_ACCESS_REGEX};

lazy_static! {
    /// a solidity type name, the only identifier a cast around a storage read introduces
    static ref TYPE_NAME_REGEX: Regex =
        Regex::new(r"^(u?int\d*|address|bytes\d*|bool|string)$").expect("failed to build regex");
}

/// The return line of a getter's logic: the single `return`, when every other line is a check
/// the compiler puts before it (on msg.value and the calldata size, never on state) or the
/// scratch memory it fills (vyper copies the selector, and the value read, to memory). A function
/// that branches, checks state, writes, calls or returns on several paths (a cached value on
/// one, a computed one on another) has none, whatever the analyzer saw one of its paths return.
pub(crate) fn getter_return_line(logic: &[String]) -> Option<&String> {
    let plain = logic.iter().all(|line| {
        let line = line.trim_start();
        line.starts_with("return") ||
            (line.starts_with("require(") &&
                !line.contains("storage[") &&
                !line.contains("transient[")) ||
            SCRATCH_ASSIGNMENT_REGEX.is_match(line).unwrap_or(false)
    });
    let mut returns = logic.iter().filter(|line| line.trim_start().starts_with("return"));
    match (plain, returns.next(), returns.next()) {
        (true, Some(line), None) => Some(line),
        _ => None,
    }
}

/// The storage slot `expr` reads the way a public getter does, i.e. `expr` is a single storage
/// access that is at most cast, masked (`&`), shifted (`>>`, or `/ 0x0100` from older solc) or
/// normalised to a bool (`!(!x)`) out of a packed slot. A value computed from storage
/// (`storage[0x05] + 0x01`, `address(this).balance - storage[0x05]`, two reads, a second return
/// word) is not a getter's return, and the function keeps its logic.
pub(crate) fn getter_storage_slot(expr: &str) -> Option<String> {
    let access = STORAGE_ACCESS_REGEX.find(expr).ok().flatten()?.as_str();
    let range = find_balanced_encapsulator(access, ('[', ']')).ok()?;
    let slot = format!("storage[{}]", &access[range]);

    // `string(rlp.encodePacked(storage[..]))` is how the postprocessor writes a string getter
    let rest = expr.replacen(&slot, "", 1).replace("rlp.encodePacked", "");
    if rest.contains("storage") {
        return None;
    }

    // what is left may only be `return`, casts, parentheses, masks, shifts, negations and
    // constants
    let plain_chars = rest.chars().all(|c| {
        c.is_ascii_alphanumeric() ||
            matches!(c, '_' | ' ' | '(' | ')' | '&' | '>' | '/' | '!' | ';')
    });
    let plain_tokens = rest
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|token| !token.is_empty())
        .all(|token| {
            token == "return" ||
                TYPE_NAME_REGEX.is_match(token).unwrap_or(false) ||
                token.starts_with("0x") ||
                token.chars().all(|c| c.is_ascii_digit())
        });

    (plain_chars && plain_tokens).then_some(slot)
}

#[cfg(test)]
mod tests {
    use super::{getter_return_line, getter_storage_slot};

    fn logic(lines: &[&str]) -> Vec<String> {
        lines.iter().map(|line| line.to_string()).collect()
    }

    #[test]
    fn test_getter_return_line_accepts_checks_and_scratch_memory() {
        let solc = logic(&["require(msg.value);", "return storage[0x05];"]);
        assert_eq!(getter_return_line(&solc).map(String::as_str), Some("return storage[0x05];"));

        // vyper: the selector and the value read go through memory first
        let vyper = logic(&[
            "memory[0x00] = msg.data[0] >> 0xe0;",
            "require(msg.value);",
            "var_a = storage[0x05];",
            "return storage[0x05];",
        ]);
        assert!(getter_return_line(&vyper).is_some());
    }

    #[test]
    fn test_getter_return_line_rejects_logic() {
        for lines in [
            // a check on state is logic (a float computation written as overflow checks)
            vec!["require(0 - storage[0x0c], \"float128: overflow\");", "return storage[0x05];"],
            // a cached value on one path, a computed one on another
            vec![
                "if (storage[0x05]) {",
                "return storage[0x05];",
                "}",
                "return storage[0x06] * 0x02;",
            ],
            vec![
                "(bool success, bytes memory ret0) = address(arg0).call(var_a);",
                "return storage[0x05];",
            ],
            vec!["storage[0x05] = 0x01;", "return storage[0x05];"],
            vec!["emit Event_abcdef01();", "return storage[0x05];"],
        ] {
            assert!(getter_return_line(&logic(&lines)).is_none(), "{lines:?}");
        }
    }

    #[test]
    fn test_getter_storage_slot_accepts_plain_reads() {
        for expr in [
            "storage[0x05]",
            "return storage[0x05];",
            "(address(storage[0x05]))",
            "return (0xffffffffffffffffffffffffffffffffffffffff & storage[0x05]);",
            "(uint8((storage[0x00] >> 0x08) & 0xff))",
            "return address(storage[0x00] / 0x0100);",
            "return !(!bytes1(storage[0x00] >> 0x08));",
            "return string(rlp.encodePacked(storage[0x03]));",
            "storage[keccak256(0x00, 0x40)]",
        ] {
            assert!(getter_storage_slot(expr).is_some(), "{expr}");
        }
        assert_eq!(
            getter_storage_slot("return storage[keccak256(0x00, 0x40)];").as_deref(),
            Some("storage[keccak256(0x00, 0x40)]")
        );
    }

    #[test]
    fn test_getter_storage_slot_rejects_computed_values() {
        for expr in [
            "return storage[0x05] + 0x01;",
            "((address(this).balance - storage[0x05]) * 0x14) / 0x2710",
            "storage[0x05], storage[0x06]",
            "0x20, memory[0x40].length",
            "!(storage[0x05] == 0x02)",
            "storage[0x05] | arg0",
        ] {
            assert!(getter_storage_slot(expr).is_none(), "{expr}");
        }
    }
}
