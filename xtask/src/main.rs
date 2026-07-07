use std::{
    env,
    path::{Path, PathBuf},
    process::{self, Command},
    sync::LazyLock,
};

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    action: Action,
}

#[derive(Debug, Subcommand)]
enum Action {
    Dev,
    DevOpt,
}

static WORKSPACE_PATH: LazyLock<PathBuf> = LazyLock::new(|| {
    Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .to_path_buf()
});

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.action {
        Action::Dev => dev()?,
        Action::DevOpt => dev_opt()?,
    };

    Ok(())
}

fn dev() -> Result<()> {
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());

    let status = Command::new(cargo)
        .current_dir(&*WORKSPACE_PATH)
        .args(&[
            "run",
            "--package",
            "isoplot-bevy",
            "-F",
            "bevy/dynamic_linking",
        ])
        .status()
        .expect("Failed to execute cargo run");

    if !status.success() {
        process::exit(status.code().unwrap_or(1));
    }

    Ok(())
}

fn dev_opt() -> Result<()> {
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());

    let status = Command::new(cargo)
        .current_dir(&*WORKSPACE_PATH)
        .args(&[
            "run",
            "--package",
            "isoplot-bevy",
            "--release",
            "-F",
            "bevy/dynamic_linking",
        ])
        .status()
        .expect("Failed to execute cargo run");

    if !status.success() {
        process::exit(status.code().unwrap_or(1));
    }

    Ok(())
}
