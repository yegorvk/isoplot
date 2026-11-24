use isoplot_expr::{Artifact, FieldType, Shape};

fn main() {
    let artifact: Artifact<[f32; 3], f32> = Artifact::compile(
        &Shape {
            params: &[
                ("x", FieldType::F32),
                ("y", FieldType::F32),
                ("z", FieldType::F32),
            ],
        },
        "y - ln(x^2 - z^2)",
    )
    .unwrap();

    println!("{:#?}", artifact.autodiff());
}
