//! Regression tests for function discovery: every selector a runtime's dispatcher answers gets a
//! function (or, for a getter, a public storage variable) in the decompile, whatever form the
//! compiler gave the compare. The runtimes are in `testdata/dispatch/` and run offline.

#[cfg(test)]
mod dispatcher_tests {
    use std::path::PathBuf;

    use heimdall_decompiler::{decompile, DecompilerArgsBuilder};

    /// Decompiles a fixture on a runtime with a large thread stack: the symbolic walk recurses
    /// once per branch, and a debug build overflows the default stack on the larger runtimes.
    fn decompile_fixture(name: &str) -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("testdata")
            .join("dispatch")
            .join(format!("{name}.hex"));
        let bytecode = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
            .trim()
            .to_string();

        let args = DecompilerArgsBuilder::new()
            .target(bytecode)
            .skip_resolving(true)
            .include_solidity(true)
            .timeout(60000)
            .build()
            .expect("failed to build args");

        tokio::runtime::Builder::new_multi_thread()
            .thread_stack_size(512 << 20)
            .enable_all()
            .build()
            .expect("failed to build runtime")
            .block_on(async {
                tokio::task::spawn(decompile(args))
                    .await
                    .expect("decompile task panicked")
                    .unwrap_or_else(|e| panic!("failed to decompile {name}: {e}"))
            })
            .source
            .expect("decompile source is empty")
            .to_lowercase()
    }

    /// The selectors with no `Unresolved_<selector>` function and no `unresolved_<selector>`
    /// storage variable (a getter is folded into a public variable) in the source.
    fn missing<'a>(source: &str, selectors: &[&'a str]) -> Vec<&'a str> {
        selectors
            .iter()
            .copied()
            .filter(|selector| !source.contains(&format!("unresolved_{selector}")))
            .collect()
    }

    #[test]
    fn test_dispatcher_solc_sub_last_leaf_and_fallthrough_body() {
        // solc 0.8 via-IR flash-loan executor (polygon 0xc2c6f4f7e1425be1dcb996175cf7852bf53c23e2
        // at block 95204744): seven EQ leaves, and the eighth selector compiled as the last leaf
        // `PUSH4 f04f2707 SUB PUSH2 000f JUMPI` with the body falling through right after the
        // JUMPI.
        let source = decompile_fixture("flash-loan-executor-solc-0.8");
        let selectors = [
            "610c95d0", "6c0137c4", "91dd7346", "de463630", "e352d1b0", "e56e7158", "e9cbafb0",
            "f04f2707",
        ];
        assert_eq!(missing(&source, &selectors), Vec::<&str>::new());
        assert!(source.contains("function unresolved_f04f2707("));

        // the calls the flash-loan callbacks make on the vault, 0x33cfcd3b and 0x7ff00809, sit
        // behind a shared memory-allocation helper; a body that lost them says "flash loan, swap,
        // settle" and nothing else
        for call in ["33cfcd3b", "7ff00809"] {
            assert!(source.contains(call), "call site 0x{call} missing from every body");
        }
    }

    #[test]
    fn test_dispatcher_solc_short_push_selector() {
        // Uniswap v4 PoolManager (solc 0.8.26): a binary-search table with GT pivots, and
        // balanceOf(address,uint256) = 0x00fdd58e compiled as `PUSH3 fdd58e EQ`.
        let source = decompile_fixture("uniswap-v4-poolmanager-solc-0.8.26");
        let selectors = [
            "00fdd58e", "01ffc9a7", "095bcdb6", "0b0d9c09", "11da60b4", "156e29f6", "1e2eaeaf",
            "234266d7", "2d771389", "35fd631a", "3dd45adb", "426a8493", "48c89491", "52759651",
            "558a7297", "598af9e7", "5a6bcfda", "6276cbbe", "7e87ce7d", "80f0b44c", "8161b874",
            "8da5cb5b", "97e8cd4e", "9bf6645f", "a5841194", "b6363cf2", "dbd035ff", "f02de3b2",
            "f135baaa", "f2fde38b", "f3cd914c", "f5298aca", "fe99049a",
        ];
        assert_eq!(missing(&source, &selectors), Vec::<&str>::new());
    }

    #[test]
    fn test_dispatcher_vyper_0_2_mload_eq_iszero() {
        // Curve 3pool (vyper 0.2.4): `PUSH4 sel PUSH1 00 MLOAD EQ ISZERO PUSH2 next JUMPI`, each
        // leaf jumping over its body, the table interleaved with the bodies.
        let source = decompile_fixture("curve3pool-vyper-0.2.4");
        let selectors = [
            "07211ef7", "14052288", "1a4d01d2", "1ec0cdc1", "2081066c", "226840fb", "3046f972",
            "30c54085", "3883e119", "3c157e64", "3df02124", "405e28f8", "4515cef3", "4903b0d1",
            "4f12fe97", "524c3901", "5409491a", "551a6588", "58680d0b", "5b5a1467", "5e0d443f",
            "6a1c05ae", "6b441a40", "86fbf193", "8da5cb5b", "9fdaea0c", "b4b577ad", "bb7b8b80",
            "c6610657", "cc2b27d7", "ddca3f43", "e0a0b586", "e2e7d264", "e3698853", "e3824462",
            "ecb586a5", "f446c1d0", "fee3f7f9",
        ];
        assert_eq!(missing(&source, &selectors), Vec::<&str>::new());
    }

    #[test]
    fn test_dispatcher_vyper_0_2_short_push_selector() {
        // veCRV (vyper 0.2.4): the same table with balanceOf(address,uint256) = 0x00fdd58e as a
        // PUSH3.
        let source = decompile_fixture("vecrv-vyper-0.2.4");
        let selectors = [
            "00fdd58e", "010ae757", "047fc9aa", "06fdde03", "17f7182a", "18160ddd", "28d09d47",
            "313ce567", "3a46273e", "3ccfd60b", "3cebb823", "4957677c", "4ee2cd7e", "54fd4d50",
            "57f901e2", "65fc3873", "6a1c05ae", "6b441a40", "70a08231", "71197484", "7175d4f7",
            "7c74a174", "8e5b490f", "8ff36fd1", "900cf0cf", "95d89b41", "981b24d0", "adc63589",
            "bd85b039", "bef97c87", "c2c4c5c1", "cbf9fe5f", "d1febfb9", "da020a18", "eff7a612",
            "f77c4791", "f851a440", "fc0c546a",
        ];
        assert_eq!(missing(&source, &selectors), Vec::<&str>::new());
    }

    #[test]
    fn test_dispatcher_vyper_0_3_xor() {
        // crvUSD controller (vyper 0.3.7): `PUSH4 sel DUP2 XOR PUSH2 next JUMPI`, with
        // 0x007c98ab as a PUSH3.
        let source = decompile_fixture("crvusd-controller-vyper-0.3.7");
        let selectors = [
            "007c98ab", "036aed88", "0b8db681", "152f65cb", "1b1800e3", "1b25cdaf", "1e0cfcef",
            "22c71453", "23cfed03", "24049e57", "2621db2f", "2a0c3586", "2a943945", "2c5089c3",
            "2e4af52a", "31dc3ca8", "371fd8e6", "37671f93", "3ecdb828", "4189617d", "4f02c420",
            "5449b9cb", "5457ff7b", "546e040d", "627d2b83", "6cce39be", "6f972f12", "7128f3b8",
            "720fb254", "80e8f6ec", "81d2f1b7", "8908ea82", "90f8667d", "9a497196", "9b6c56ec",
            "a21adb9e", "a5b4804a", "a7573206", "acb70815", "adfae4ce", "b4440df4", "bc61ea23",
            "bcbaf487", "c45a0155", "cc1891c7", "d14ff5b6", "d9f11a64", "dd171e7c", "e1ec3c68",
            "e231bff0", "e2d8ebee", "ec74d0a8",
        ];
        assert_eq!(missing(&source, &selectors), Vec::<&str>::new());
    }
}
