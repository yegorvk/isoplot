use isoplot_expr::{Diagnostic, FieldType, Shape, dump_ast};
use std::{env, process::ExitCode};

fn main() -> ExitCode {
    let source: Vec<String> = env::args().skip(1).collect();
    let source = source.join(" ");

    if source.is_empty() {
        eprintln!("usage: cli_ast <expression>");
        return ExitCode::from(2);
    }

    let (parsed, diagnostics) = dump_ast::<[f32; 3]>(
        &Shape {
            params: &[
                ("x", FieldType::F32),
                ("y", FieldType::F32),
                ("z", FieldType::F32),
            ],
        },
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
