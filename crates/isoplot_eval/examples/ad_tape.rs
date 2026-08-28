use isoplot_eval::{Program, ProgramDesc};

fn main() {
    let program: Program<[f32; 3], f32> = Program::compile(
        &ProgramDesc::new(&["x", "y", "z"], &[]),
        "y - ln(x^2 - z^2)",
    )
    .unwrap();

    println!("{:#?}", program.autodiff());
}
