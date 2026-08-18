use std::{env, process::ExitCode};

use isoplot_eval::{Diagnostic, ProgramDesc, dump_ast};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: cli_ast <expression>");
        return ExitCode::from(2);
    }
    let source = args.join(" ");

    let (parsed, diagnostics) = dump_ast(
        &ProgramDesc::<[f32; 3]>::new(&["x", "y", "z"], &[]),
        &source,
    );
    println!("{}", parsed.pretty_printer());

    for diagnostic in &diagnostics {
        println!();
        render(&source, diagnostic);
    }

    if diagnostics.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn render(source: &str, diagnostic: &Diagnostic) {
    let span = diagnostic.location();
    let (start, end) = (span.start.index(), span.end.index());
    let offset = source[..start].chars().count();
    let width = source[start..end].chars().count().max(1);

    println!("error: {}", diagnostic.message());
    println!("  {source}");
    println!("  {}{}", " ".repeat(offset), "^".repeat(width));
}
