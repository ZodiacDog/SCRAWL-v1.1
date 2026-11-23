use scrawl::opcodes::{all_base_opcodes, Opcode};
use scrawl::vm::{implemented_opcodes, unimplemented_opcodes};

#[test]
fn base_isa_is_84_opcodes() {
    assert_eq!(all_base_opcodes().len(), 84);
}

#[test]
fn implemented_plus_unimplemented_is_84_base() {
    let impld: Vec<Opcode> = implemented_opcodes()
        .into_iter()
        .filter(|o| !matches!(o, Opcode::Lexicon(_)))
        .collect();
    let unimpl = unimplemented_opcodes();
    assert_eq!(impld.len() + unimpl.len(), 84, "impl {} unimpl {}", impld.len(), unimpl.len());
}

#[test]
fn lex_call_is_implemented_but_not_a_base_opcode() {
    assert!(implemented_opcodes().iter().any(|o| matches!(o, Opcode::Lexicon(_))));
    assert!(all_base_opcodes().iter().all(|o| !matches!(o, Opcode::Lexicon(_))));
}
