mod controls;
mod plot;

use std::{collections::HashMap, f32::consts::PI};

use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    app::TaskPoolThreadAssignmentPolicy,
    input_focus::{AutoFocus, FocusCause, InputFocus},
    prelude::*,
    text::{EditableText, TextCursorStyle},
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};

use isoplot_eval::{
    CompileError, DefaultBackend, DefaultInstance, DefaultMultiBackend, DefaultMultiInstance,
    Diagnostic, Evaluator, Program, ProgramDesc, ProgramShape,
};
use isoplot_mesh::{NormalField, ScalarField};

use crate::{
    controls::{CameraControls, CameraControlsPlugin},
    plot::{Plot, PlotPlugin, PlotSource},
};

struct Equation {
    scalar_sampler: DefaultInstance,
    normal_sampler: DefaultMultiInstance,
}

impl Equation {
    fn new(equation: &str) -> Result<Self, CompileError> {
        let program = Program::compile(
            &ProgramDesc::new(&equation_default_shape(), HashMap::new()),
            equation,
        )?;

        Ok(Self {
            normal_sampler: program.autodiff().instantiate(),
            scalar_sampler: program.instantiate(),
        })
    }

    fn create_source(&self) -> DynamicSource {
        DynamicSource {
            evaluator: self.scalar_sampler.evaluator(),
            gradient: self.normal_sampler.evaluator(),
        }
    }
}

impl PlotSource for Equation {
    fn field(&self) -> impl NormalField {
        self.create_source()
    }
}

fn equation_default_shape() -> ProgramShape {
    ProgramShape::builder()
        .with_input("x")
        .with_input("y")
        .with_input("z")
        .build()
}

struct DynamicSource {
    evaluator: Evaluator<DefaultBackend>,
    gradient: Evaluator<DefaultMultiBackend>,
}

impl ScalarField for DynamicSource {
    fn sample(&self, point: Vec3) -> f32 {
        self.evaluator.evaluate(&point.to_array())
    }

    fn is_flat(&self, min: Vec3, size: f32) -> bool {
        let mut samples = [(0.0f32, Vec3::ZERO); 9];

        for (i, sample) in samples.iter_mut().enumerate() {
            let offset = if i < 8 {
                Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32)
            } else {
                Vec3::splat(0.5)
            };

            *sample = self.sample_with_gradient(min + size * offset);
        }

        let has_negative = samples.iter().any(|(v, _)| v.is_sign_negative());
        let has_positive = samples.iter().any(|(v, _)| v.is_sign_positive());

        if !(has_negative && has_positive) {
            let min_abs = samples
                .iter()
                .map(|(v, _)| v.abs())
                .fold(f32::INFINITY, f32::min);
            let max_slope = samples.iter().map(|(_, g)| g.length()).fold(0.0, f32::max);
            return min_abs > 2.0 * max_slope * size * 3f32.sqrt();
        }

        let mean_normal = samples
            .iter()
            .map(|(_, g)| g.normalize_or_zero())
            .sum::<Vec3>()
            .normalize_or_zero();

        samples
            .iter()
            .all(|(_, g)| g.normalize_or_zero().dot(mean_normal) > 0.995)
    }
}

impl DynamicSource {
    fn sample_with_gradient(&self, point: Vec3) -> (f32, Vec3) {
        let mut outputs = [0.0f32; 4];
        self.gradient.evaluate_into(&point.to_array(), &mut outputs);
        (outputs[0], Vec3::new(outputs[1], outputs[2], outputs[3]))
    }
}

impl NormalField for DynamicSource {
    fn sample_normal(&self, point: Vec3) -> Vec3 {
        self.sample_with_gradient(point).1.normalize_or_zero()
    }
}

const DEFAULT_EQUATION: &str = "max(y - (x^2 + z^2), x^2+y^2+z^2 - 10)";

#[derive(Resource)]
struct ActivePlot {
    entity: Entity,
    material: Handle<StandardMaterial>,
}

fn main() {
    App::new()
        .add_systems(Startup, setup)
        .add_systems(Update, (toggle_focus, submit_equation))
        .add_plugins((
            DefaultPlugins.set(TaskPoolPlugin {
                task_pool_options: TaskPoolOptions {
                    async_compute: TaskPoolThreadAssignmentPolicy {
                        min_threads: 1,
                        max_threads: usize::MAX,
                        percent: 1.0,
                        on_thread_spawn: None,
                        on_thread_destroy: None,
                    },
                    ..default()
                },
            }),
            PlotPlugin,
            CameraControlsPlugin,
        ))
        .run();
}

fn setup(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    let simple_material = materials.add(StandardMaterial {
        base_color: Color::LinearRgba(LinearRgba::GREEN),
        metallic: 1.0,
        cull_mode: None,
        double_sided: true,
        ..default()
    });

    let plot = create_plot(Equation::new(DEFAULT_EQUATION).unwrap());
    let entity = spawn_plot(&mut commands, plot, simple_material.clone());

    commands.insert_resource(ActivePlot {
        entity,
        material: simple_material,
    });

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform {
            translation: Vec3::ZERO,
            rotation: Quat::from_rotation_x(-0.4 * PI),
            ..default()
        },
    ));

    commands.spawn((
        EquationText,
        EditableText {
            visible_width: Some(40.0),
            ..EditableText::new(DEFAULT_EQUATION)
        },
        TextCursorStyle::default(),
        AutoFocus,
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(8.0),
            padding: UiRect::all(Val::Px(4.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
    ));

    commands.spawn((
        DiagnosticsText,
        Text::new(""),
        TextColor(Color::srgb(1.0, 0.5, 0.5)),
        Visibility::Hidden,
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(48.0),
            left: Val::Px(8.0),
            padding: UiRect::all(Val::Px(4.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.4, 1.0).looking_to(Dir3::NEG_Z, Dir3::Y),
        CameraControls::default(),
        TemporalAntiAliasing { reset: false },
        Msaa::Off,
        AmbientLight {
            brightness: 120.0,
            ..default()
        },
    ));
}

fn spawn_plot(commands: &mut Commands, plot: Plot, material: Handle<StandardMaterial>) -> Entity {
    commands
        .spawn((
            plot,
            MeshMaterial3d(material),
            Transform::from_scale(Vec3::splat(3.0)),
        ))
        .id()
}

#[derive(Component)]
struct EquationText;

#[derive(Component)]
struct DiagnosticsText;

fn submit_equation(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut active: ResMut<ActivePlot>,
    field: Single<(&EditableText, &mut TextColor), With<EquationText>>,
    overlay: Single<(&mut Text, &mut Visibility), With<DiagnosticsText>>,
) {
    if !keys.just_pressed(KeyCode::Enter) {
        return;
    }

    let (field, mut color) = field.into_inner();
    let (mut overlay_text, mut overlay_visibility) = overlay.into_inner();
    let source = field.value().to_string();

    let equation = match Equation::new(&source) {
        Ok(equation) => equation,
        Err(error) => {
            color.0 = Color::srgb(1.0, 0.3, 0.3);
            overlay_text.0 = render_diagnostics(&source, error.diagnostics());
            *overlay_visibility = Visibility::Visible;
            return;
        }
    };

    color.0 = Color::WHITE;
    *overlay_visibility = Visibility::Hidden;
    commands.entity(active.entity).despawn();
    active.entity = spawn_plot(
        &mut commands,
        create_plot(equation),
        active.material.clone(),
    );
}

fn render_diagnostics(source: &str, diagnostics: &[Diagnostic]) -> String {
    let mut out = String::new();

    for (i, diagnostic) in diagnostics.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }

        let span = diagnostic.location();
        let (start, end) = (span.start.index(), span.end.index());
        let offset = source[..start].chars().count();
        let width = source[start..end].chars().count().max(1);

        out.push_str(&format!(
            "error: {}\n  {source}\n  {}{}",
            diagnostic.message(),
            " ".repeat(offset),
            "^".repeat(width),
        ));
    }

    out
}

fn create_plot(equation: Equation) -> Plot {
    Plot::new(equation, 5, 7)
}

fn toggle_focus(
    mouse: Res<ButtonInput<MouseButton>>,
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut controls: Query<&mut CameraControls>,
    mut input_focus: ResMut<InputFocus>,
    expr_field: Single<Entity, With<EquationText>>,
) {
    if mouse.just_pressed(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
        input_focus.clear();

        for mut controls in &mut controls {
            controls.is_active = true;
        }
    }

    if mouse.just_released(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
        input_focus.set(*expr_field, FocusCause::Navigated);

        for mut controls in &mut controls {
            controls.is_active = false;
        }
    }
}
